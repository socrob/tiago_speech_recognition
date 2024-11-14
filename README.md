# TIAGo Speech Recognition

## Overview

The TIAGo Speech Recognition package is responsible for enabling speech recognition capabilities on the TIAGo robot. This package uses advanced speech recognition models to interpret user instructions and convert them into actionable commands for the robot.

## Features
- **Advanced Speech Recognition**: Utilizes state-of-the-art models to accurately transcribe spoken language into text.
- **Configurable Search Algorithms**: Supports various search algorithms like beam search and diverse beam search for improved recognition accuracy.
- **Error Handling**: Includes mechanisms to handle common ASR (Automatic Speech Recognition) errors.
- **ROS Integration**: Seamlessly integrates with ROS, allowing easy communication with other ROS nodes.

## Requirements

- ROS version: Noetic
- Dependencies:
  - [transformers](https://github.com/huggingface/transformers)
  - [speech_recognition](https://pypi.org/project/SpeechRecognition/)
  - [socrob_speech_msgs](https://github.com/socrob/socrob_speech_msgs) 
  - [audio_common](https://wiki.ros.org/audio_common)

## Installation

### 0. Install the message modules
Follow the installation instructions in the [socrob_speech_msgs](https://github.com/socrob/socrob_speech_msgs). Then install the `audio_common` package with:

```bash
sudo apt-get install ros-noetic-audio-common
```

### 1. Clone the repository
```bash
cd ~/<your_workspace>/src
git clone https://github.com/certafonso/tiago_speech_recognition.git
```

### 2. Install dependencies

Navigate to the cloned repository and install the required dependencies:

```bash
cd tiago_speech_recognition
pip install -r requirements.txt
```

### 3. Build the workspace
Navigate to your catkin workspace and build the package:

```bash
cd ~/<your_workspace>
catkin build
```

### 4. Source the setup file
After building, source the workspace to update the environment:

```bash
source ~/<your_workspace>/devel/setup.bash
```

## Usage

### Launching the Node

To launch only the speech recognition node, use the following command:

```bash
roslaunch tiago_speech_recognition_ros speech_recognition_node.launch
```

#### Launch File Arguments

The launch file `speech_recognition_node.launch` accepts several arguments to customize the behavior of the speech recognition node:

- `silence_level`: The level of silence used to determine when to stop recording. Default is `300`.
- `energy_threshold_ratio`: This parameter defines the ratio used to determine the energy threshold for speech detection. For example, if the `silence_level` is set to 100 and the `energy_threshold_ratio` is 1.5, the recording will stop when the energy level drops below 150 (i.e., 100 * 1.5). The default value is `1.5`.
- `model`: The ID of the speech recognition model from the Hugging Face hub. Default is `openai/whisper-small.en`.
- `save_wav`: If set to `true`, the system will save debug WAV files. Default is `false`.
- `node_name`: The name of the speech recognition node. Default is `tiago_speech_recognition`.
- `transcript_topic`: The topic name where the transcribed text will be published. Default is `~transcript`.
- `audio_topic`: The topic name where the audio data will be published. Default is `/microphone_node/audio`.
- `generation_config`: Path to the ASR generation configuration file. Defaults to beam search.

These arguments allow you to fine-tune the speech recognition node's behavior to match your specific requirements and environment.

### Launching the Speech Pipeline

To launch the entire SocRob speech pipeline use:

```bash
roslaunch tiago_speech_recognition_ros tiago_speech_recognition.launch
```

This will launch the speech recognition node, the [microphone](https://github.com/socrob/microphone_node) node and [keyword recognition](https://github.com/socrob/keyword_detection) node.

#### Launch File Arguments

The launch file `tiago_speech_recognition.launch` accepts several arguments to customize the behavior of the speech recognition system:

- `microphone_device`: Specifies the microphone device to be used. Default is `"default"` which will use the default microphone of the computer.
- `launch_mic`: Determines whether to launch the microphone node. Default is `true`.
- `launch_keyword`: Determines whether to launch the keyword detection node. Default is `true`.
- `save_wav`: If set to `true`, the system will save debug WAV files. Default is `true`.
- `asr_node_name`: The name of the ASR (Automatic Speech Recognition) node. Default is `"tiago_speech_recognition"`.
- `ASR_generation_config`: Path to the ASR generation configuration file. Defaults to beam search.
- `transcript_topic`: The topic name where the transcribed text will be published. Default is `"/tiago_speech_recognition/transcript"`.
- `audio_topic`: The topic name where the audio data will be published. Default is `"/microphone_node/audio"`.

These arguments allow you to tailor the speech recognition setup to your specific needs and hardware configuration.

### Configuration Files

The decoding method for the transformer can be personalized with custom YAML files. This corresponds to setting keyword arguments in hugging face's `model.generate` method.

The following config files are included in the `config` folder:

- `beam_search.yaml`: Configuration for beam search algorithm.
- `diverse_beam_search.yaml`: Configuration for diverse beam search algorithm.

## Common ASR Errors

When the node is initialized it will load a list of common ASR errors loaded in the `common_asr_errors.yaml` file in the config folder. This information will be published in two parameters so it can be used in upstream modules to potentially correct these errors:

- `tiago_speech_recognition/common_asr_errors`: Will contain a dictionary mapping each word to every defined possible misspleling. See example:
```yaml
pear: [bear]
crisps: [crisp]
pringles: [pringle]
tictac: [tic tac]
```

- `tiago_speech_recognition/common_asr_errors_categorized`: Will contain the full data present in the `common_asr_errors.yaml` file, which is the same as the `common_asr_errors`, but divided into categories. See example:
```yaml
fruits:
  pear: [bear]
snacks:
  crisps: [crisp]
  pringles: [pringle]
  tictac: [tic tac]
```
