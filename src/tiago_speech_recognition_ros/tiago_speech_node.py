#!/usr/bin/env python3

import speech_recognition as sr
import rospy
import rospkg

# Import service messages and data types
from std_msgs.msg import String
from socrob_speech_msgs.msg import ASRNBestList, ASRHypothesis
from audio_common_msgs.msg import AudioData

# from actions_tiago_ros.tiago_api import TiagoAPI
# tiago_api = TiagoAPI()

from my_robot_common.modules.common import getFilePathFromRospkg
import tiago_speech_recognition_ros.whisper as asr_engine

# ros independent imports
from datetime import datetime
import pyaudio
import audioop
from threading import Thread
from queue import Queue, Empty
import torch.cuda
from yaml import safe_load
import gc

class TiagoASR():
    def __init__(self):
        rospy.init_node('tiago_speech_recognition')

        # Get model
        if rospy.has_param("~model_id"):
            self.model_id = rospy.get_param("~model_id")
        else:
            self.model_id = "openai/whisper-small.en"
            rospy.logwarn(f"Model id not set. Using default value of {self.model_id}")
        
        rospy.loginfo(f"Using ASR model: {self.model_id}")

        # get an instance of RosPack with the default search paths
        self.rospack = rospkg.RosPack()

        #Publishers
        self.output_pub = rospy.Publisher(rospy.get_param("~transcript_topic", "~transcript"), 
                                          ASRNBestList, queue_size = 5)

        #Subscriptions
        self.event_sub = rospy.Subscriber('~event_in', String, self.event_in_callback)
        self.audio_sub = rospy.Subscriber(rospy.get_param("~audio_topic", "/microphone_node/audio"),
                                          AudioData, self.audio_callback)

        self.rate = rospy.Rate(2)

        self.listening = False
        self.recording = False
        self.loading_model = False

        # Pipeline for ASR
        self.pipeline = None

        # This param notifies other nodes if this one is recording or if they can stop the microphone
        rospy.set_param("~recording", False)

        # The device to load the model to
        self.device = rospy.get_param("~device", "cuda:0" if torch.cuda.is_available() else "cpu")

        # Thread for recording audio
        self.recording_thread = None

        # Buffers for audio data and recordings
        self.audio_buffer = Queue(maxsize=2)
        self.recordings_buffer = Queue()
        self.audio_bytes_buffer = []

        rospy.loginfo("ASR interface initialized")

        # Start the node in the correct mode
        self.event_in_callback(String(rospy.get_param("~startup_event", "e_start")))

    def main(self):
        while not rospy.is_shutdown():
            # Try to read audio from the buffer on a timeout
            try:
                audio = self.recordings_buffer.get(block=True, timeout=1)
            except Empty:
                continue

            # When a recording arrives to the buffer process it and publish the result
            rospy.loginfo("ASR received audio")
            message_list = self.speech_recognizer(audio)

            if message_list is not None:
                self.output_pub.publish(message_list)

        # Make sure other threads are stopped before finishing
        self.listening = False
        self.recording = False
            
    def event_in_callback(self, msg):
        command = msg.data.split("_")
        if command[0] != "e":
            return

        # A recording event, formated as "e_record_{silence_seconds}" where {silence_seconds}
        # is the number of seconds the recording will run for if no sound is detected and is 
        # an optional parameter
        if command[1] == "record" and self.listening:
            if len(command) == 3:
                self.max_wait = float(command[2])

            else:
                self.max_wait = 1

            self.recording = True

        # e_stop will stop all recordings and unload the model from memory
        elif command[1] == "stop" and self.listening:
            self.recording = False
            self.listening = False

            self.recording_thread.join()

            # Remove the model from memory
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

            rospy.loginfo("Stopped listenning")

        # e_start will load the model
        elif command[1] == "start" and not self.listening and not self.loading_model:
            self.loading_model = True
            
            # Wait 5 seconds for a microphone because we need for it to publish the recording properties
            time_start = rospy.Time.now()
            timeout_duration = rospy.Duration(secs=5)
            while self.audio_sub.get_num_connections() == 0:
                # Timeout
                if rospy.Time.now() - time_start > timeout_duration:
                    rospy.logerr("No microphone node detected, did you launch it?")
                    self.loading_model = False
                    return
                
                rospy.sleep(0.5)

            # Get audio recording properties from parameter server
            if rospy.has_param("/microphone_node/sample_rate"):
                self.sample_rate = rospy.get_param("/microphone_node/sample_rate")
            else:
                self.sample_rate = 16000
                rospy.logwarn(f"Sample rate not set. Using default value of {self.sample_rate}")

            if rospy.has_param("/microphone_node/frame_length"):
                self.frame_length = rospy.get_param("/microphone_node/frame_length")
            else:
                self.frame_length = 512
                rospy.logwarn(f"Frame lenght not set. Using default value of {self.frame_length}")
                
            rospy.logdebug(f"Microphone settings: sample rate={self.sample_rate}, frame lenght={self.frame_length}")

            self.seconds_per_frame = self.frame_length / self.sample_rate
            
            # Check if microphone is recording
            if rospy.has_param("/microphone_node/recording"):
                if not rospy.get_param("/microphone_node/recording"):
                    rospy.logwarn("Microphone is not recording")

            rospy.loginfo(f"Loading ASR model '{self.model_id}'...")

            # Create a pipeline for ASR
            self.model = asr_engine.load_model(self.model_id, device=self.device)

            rospy.loginfo(f"Loaded model!")

            # Load generation config (defaults to greedy decoding)
            config_file = rospy.get_param("~generation_config", None)
            
            if config_file is None:
                self.gen_config = {}

            else:
                with open(config_file, "r") as f:
                    self.gen_config = safe_load(f)

            rospy.loginfo(f"Generation config: {self.gen_config}")

            # See if debug wav files should be created
            self.save_wav = rospy.get_param("~save_wav", False)

            if self.save_wav:
                rospy.logwarn("Saving every recording as a wav file. To stop this set save_wav parameter to false.")

            # Calibrate energy thereshold for silence
            if rospy.has_param("~energy_threshold_ratio"):
                energy_threshold_ratio = rospy.get_param("~energy_threshold_ratio")
            else:
                energy_threshold_ratio = 1.3
                rospy.logwarn(f"Energy threshold not set. Using default value of {energy_threshold_ratio}")

            # Wait 5 seconds for a silence level
            time_start = rospy.Time.now()
            timeout_duration = rospy.Duration(secs=5)
            while not rospy.has_param("~silence_level"):
                # Timeout
                if rospy.Time.now() - time_start > timeout_duration:
                    rospy.logerr("The silence level is not in the parameter server.")
                    return
                
                rospy.sleep(0.5)

            silence_level = rospy.get_param("~silence_level")

            self.energy_threshold = silence_level * energy_threshold_ratio

            rospy.loginfo("Energy thereshold set to %f" % self.energy_threshold)
            
            self.listening = True
            self.loading_model = False

            # Create thread for audio recording
            self.recording_thread = Thread(target=self.recording_thread_target)
            self.recording_thread.start()

        rospy.logdebug(f"listening: {self.listening}")

    def audio_callback(self, msg):
        if self.listening:
            self.audio_buffer.put(bytes(msg.data))

    def recording_thread_target(self):
        # Lenght of a pause required to stop recording
        pause_threshold = 0.8
        frames_in_pause_threshold = pause_threshold / self.seconds_per_frame

        # Buffer to store some of the audio before recording start
        frames_before_keyword = 2 / self.seconds_per_frame
        audio_frames = []

        rospy.loginfo("Recording thread started.")

        while self.listening:
            # Listen for keyword
            try:
                frame = self.audio_buffer.get(block=True, timeout=1)
            except Empty:
                continue
            
            # Add frame to local buffer
            audio_frames.append(frame)
            if len(audio_frames) > frames_before_keyword:
                audio_frames.pop(0)

            # Started recording
            if self.recording:
                rospy.loginfo(f"Recording thread is listenning for a maximum of {self.max_wait} seconds")
                rospy.set_param("~recording", True)

                # See how many silence frames until recording is stopped
                silence_frames_until_stop = self.max_wait / self.seconds_per_frame

                # Listen until there is silence
                while self.recording and silence_frames_until_stop > 0:
                    try:
                        audio_frames.append(self.audio_buffer.get(block=True, timeout=1))
                    except Empty:
                        continue
                    
                    # Frame of silence
                    if audioop.rms(audio_frames[-1], pyaudio.get_sample_size(pyaudio.paInt16)) < self.energy_threshold:
                        silence_frames_until_stop -= 1
                    
                    # Frame with speech
                    else:
                        silence_frames_until_stop = frames_in_pause_threshold

                self.recording = False
                rospy.loginfo("Recording thread stopped listenning")
                rospy.set_param("~recording", False)

                # Remove silence frames at beggining
                while audioop.rms(audio_frames[0], pyaudio.get_sample_size(pyaudio.paInt16)) < self.energy_threshold:
                    audio_frames.pop(0)

                    if len(audio_frames) == 0:
                        break

                # If there is no audio frames continue
                if len(audio_frames) == 0:
                    rospy.logwarn("Recording thread Could not detect a sound above energy thereshold.")
                    continue

                # Remove silence frames at the end
                while audioop.rms(audio_frames[-1], pyaudio.get_sample_size(pyaudio.paInt16)) < self.energy_threshold:
                    audio_frames.pop(-1)

                # Convert to audio data to wav
                frame_data = b"".join(audio_frames)
                audio = sr.AudioData(frame_data, self.sample_rate, pyaudio.get_sample_size(pyaudio.paInt16)).get_wav_data()

                # Write audio to a WAV file
                if self.save_wav:
                    rospack = rospkg.RosPack()

                    try:
                        audio_path = rospack.get_path("tiago_speech_recognition") + \
                            datetime.now().strftime("/data/recording_%Y-%m-%d_%H-%M-%S.wav")
                        
                        with open(audio_path, "wb") as f:
                            f.write(audio)

                    except Exception as e:
                        rospy.logerr(f"Error while trying to save the log file: {e}")

                # Write audio to buffer
                self.recordings_buffer.put(audio)

                # Delete local buffer
                audio_frames = []
        
    def speech_recognizer(self, audio: bytes):
        """
        This functions selects which asr will be used and calls the corresponding function
        """
        transcript_msg = ASRNBestList()

        # Use pipeline for ASR
        transcripts, scores = asr_engine.transcribe_audio(self.model, audio,
            generation_kwargs=self.gen_config,
            sample_rate=self.sample_rate,
            device=self.device)
        
        rospy.loginfo(f"Best ASR result: {transcripts[0]}")

        transcript_msg.hypothesis = [ASRHypothesis(transcript=text, confidence=score) for text, score in zip(transcripts, scores)]

        return transcript_msg


def main():
    asr = TiagoASR()
    asr.main()

if __name__ == "__main__":
    main()