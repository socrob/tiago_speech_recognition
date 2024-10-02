from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from math import exp

from typing import Tuple, List

def load_model(model_id: str, device: str) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """Loads the model and processor needed for speech recognition

    Args:
        model_id (str): The hugging face id of the model to use or a path to the local model
        device (str): The device to load the model to ("CUDA" or "CPU")

    Returns:
        Tuple[WhisperProcessor, WhisperForConditionalGeneration]: The loaded model
    """
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.config.forced_decoder_ids = None

    return processor, model

def transcribe_audio(
    model_tuple: Tuple[WhisperProcessor, WhisperForConditionalGeneration], 
    audio: bytes,
    generation_kwargs: dict,
    sample_rate: float,
    device: str) -> Tuple[List[str], List[float]]:
    """Transcribes a piece of audio

    Args:
        model_tuple (Tuple[WhisperProcessor, WhisperForConditionalGeneration]): The model to use for transcription
        audio (bytes): The contents of an audio file
        generation_kwargs (dict): Keyword arguments to use for generation
        sample_rate (float): Sample rate of the provided audio
        device (str): The device where the model is loaded

    Returns:
        Tuple[List[str], List[float]]: The results if the transcription and respective confidence levels
    """
    
    # Unpack processor and model
    processor, model = model_tuple

    # Read the audio data and pre-process it
    inputs = ffmpeg_read(audio, sample_rate)
    input_features = processor(inputs, sampling_rate=sample_rate, return_tensors="pt").input_features

    # Make sure that the generation will output the confidence scores
    generation_kwargs.update({"output_scores": True, "return_dict_in_generate": True})

    # Transcribe the text
    output = model.generate(input_features.to(device), **generation_kwargs)
    
    # decode token ids to text
    transcription = processor.batch_decode(output.sequences, skip_special_tokens=True)

    return transcription, [exp(confidence) for confidence in output.sequences_scores]
    