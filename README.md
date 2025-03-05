# Audio Diarization with Phi-4 Multimodal Model

This notebook demonstrates how to perform audio transcription with detailed speaker diarization using the [Microsoft Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) model. The code processes audio files by generating transcriptions that include speaker labels (e.g., "Speaker 1:", "Speaker 2:") and timestamps, and it explores multiple prompt variations for diarization.

## Features

- **Dependency Installation:** Installs required libraries such as FlashAttention, PyTorch, Transformers, Accelerate, and additional utilities.
- **Model Loading:** Loads the Phi-4 multimodal model with GPU (CUDA) support and optimized attention settings.
- **Single Audio Processing:** Demonstrates diarization on one audio file with a specified prompt.
- **Batch Processing with Variations:** Iterates over multiple audio files (top 5 `.mp3` files in a specified directory) and applies 10 different prompt templates for diarization.
- **Dynamic Token Allocation:** Estimates the number of tokens required based on the audio duration to ensure the generated output fits within the model’s context.
- **Result Storage:** Saves the diarized transcription results to text files within a dedicated output directory.

## Dependencies

The notebook requires the following Python libraries:

- `flash_attn==2.7.4.post1`
- `torch==2.6.0`
- `transformers==4.48.2`
- `accelerate==1.3.0`
- `soundfile==0.13.1`
- `pillow==11.1.0`
- `scipy==1.15.2`
- `torchvision==0.21.0`
- `backoff==2.2.1`
- `peft==0.13.2`

These are installed at the beginning of the notebook using `pip`.

## Setup and Execution

1. **Notebook Environment:**  
   This code is designed to run in a Databricks Notebook environment with GPU support. The model is explicitly loaded on CUDA, and the `_attn_implementation` is set to use `flash_attention_2` for optimal performance.

2. **Model and Processor:**  
   The Phi-4 model and its associated processor are loaded from Hugging Face using the model identifier `microsoft/Phi-4-multimodal-instruct`.

3. **Audio Input:**  
   - A single audio file (e.g., `RE03a92a17d15111d9acd689a079a754f5.mp3`) is used for an initial demonstration of audio transcription with diarization.
   - A directory named `audio_files/` is used to store additional `.mp3` files. The notebook processes the top 5 audio files found in this directory.

4. **Prompt Templates:**  
   Ten different prompt templates are defined to instruct the model in various ways for speaker diarization. This allows the experiment to capture different formatting or detail levels in the generated transcriptions.

5. **Dynamic Token Calculation:**  
   For each audio file, the code estimates the required number of tokens based on the duration of the audio (taking into account an average speaking rate and adding a safety buffer). This ensures that the model generates complete and well-formatted outputs.

## Output

- **Transcription Results:**  
  Diarized transcriptions are generated for each audio file using each of the 10 prompt templates.
  
- **Result Files:**  
  The results are saved in a directory named `diarization_results` (located at the same level as the audio directory). For every audio file, a separate text file is created that includes:
  - The name of the audio file.
  - Each prompt template used.
  - The corresponding diarized transcription output.
  
## Code Structure

- **Dependency Installation:**  
  The top section installs all required libraries and restarts the Python session.

- **Model Initialization:**  
  The Phi-4 model and processor are loaded with the proper configuration for audio processing and GPU acceleration.

- **Single Audio Processing:**  
  A sample audio file is processed with a diarization-specific prompt. The prompt is constructed using special tokens (e.g., `<|user|>`, `<|assistant|>`, `<|end|>`) that follow the Phi instruct template.

- **Batch Processing with Variations:**  
  - Audio files are read from the `audio_files/` directory.
  - A helper function (`process_audio_with_template`) processes each file with a given prompt template.
  - The code loops over each audio file and each prompt template, generates the diarized transcription, and prints a preview of the response.

- **Result Saving:**  
  The final transcriptions are written to text files in the `diarization_results` directory, with clear separation of outputs corresponding to different prompt templates.
---
