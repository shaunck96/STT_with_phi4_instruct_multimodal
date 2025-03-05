# Databricks notebook source
!pip install flash_attn==2.7.4.post1
!pip install torch==2.6.0
!pip install transformers==4.48.2
!pip install accelerate==1.3.0
!pip install soundfile==0.13.1
!pip install pillow==11.1.0
!pip install scipy==1.15.2
!pip install torchvision==0.21.0
!pip install backoff==2.2.1
!pip install peft==0.13.2


# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import requests
import torch
import os
import io
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


# COMMAND ----------

# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    # if you do not use Ampere or later GPUs, change attention to "eager"
    _attn_implementation='flash_attention_2',
).cuda()


# COMMAND ----------

# Set up generation configuration
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure using phi instruct prompt template
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

# Part 1: Audio Processing with Diarization
print("\n--- AUDIO PROCESSING WITH DIARIZATION ---")
audio_path = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/audio_files/2025_01_02/RE03a92a17d15111d9acd689a079a754f5.mp3"

# Define a diarization-specific prompt using the phi instruct prompt template
diarization_prompt = (
    "Transcribe the audio to text in English with detailed speaker diarization. "
    "For each segment, label speakers as 'Speaker 1:', 'Speaker 2:', etc. "
    "If possible, include timestamps for when each speaker is speaking. "
    "Present the transcription in a structured and clear format."
)

# Combine the prompt parts to form the full prompt
prompt = f'{user_prompt}<|audio_1|>{diarization_prompt}{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

audio, samplerate = sf.read(audio_path)

# Process the audio and text input with the model's processor
inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')

# Generate the diarized transcription using the model
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)

# Remove the initial prompt tokens from the generated output
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

# Decode the generated tokens to obtain the final response
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Diarized Transcription Response\n{response}')


# COMMAND ----------

import os
import soundfile as sf
import torch
from transformers import GenerationConfig
import glob

# Set up generation configuration
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure using phi instruct prompt template
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

# Directory containing audio files
audio_dir = "/Workspace/Users/sshibu@pplweb.com/GPU_End_To_End_Code_Execution/audio_files/2025_01_02/"

# Get top 5 audio files from the directory
audio_files = glob.glob(os.path.join(audio_dir, "*.mp3"))[:5]

# Create 10 different prompt variations for diarization
prompt_templates = [
    "Transcribe the audio to text in English with detailed speaker diarization. Label speakers as 'Speaker 1:', 'Speaker 2:', etc. Include timestamps for when each speaker is speaking.",
    
    "Please provide a verbatim transcription of this audio with speaker identification. Assign numbers to different speakers (Speaker 1, Speaker 2, etc.) and include time markers in [MM:SS] format.",
    
    "Convert this audio to text while identifying different speakers. Use 'Person A:', 'Person B:', etc. for speaker labels and add timestamps in brackets when speakers change.",
    
    "Create a detailed transcript of this conversation. Distinguish between speakers using 'Speaker #' notation and indicate approximate timestamps for each speaking turn.",
    
    "Transcribe this recording with clear speaker separation. Number each speaker consistently throughout the transcript and provide timing information where possible.",
    
    "Generate a professional transcript from this audio file. Identify each unique voice as 'Voice 1', 'Voice 2', etc., and include precise timing information [HH:MM:SS].",
    
    "Perform a comprehensive diarization of this audio. Tag each speaker consistently as 'S1', 'S2', etc., and mark the start time of each speaking segment.",
    
    "Transcribe this conversation with optimal speaker differentiation. Use 'Participant 1:', 'Participant 2:', etc., and include timestamps at the beginning of each turn.",
    
    "Parse this audio into text while separating different speakers. Identify speakers as 'Speaker A', 'Speaker B', etc., and note when speaker changes occur with time references.",
    
    "Create a structured transcript of this recording with clear speaker identification. Use consistent speaker numbering and provide time markers at major transition points."
]

def process_audio_with_template(audio_path, prompt_template, processor, model):
    """Process a single audio file with a given prompt template."""
    # Construct the full prompt
    prompt = f'{user_prompt}<|audio_1|>{prompt_template}{prompt_suffix}{assistant_prompt}'
    
    # Load audio file
    audio, samplerate = sf.read(audio_path)
    
    # Calculate dynamic max_new_tokens based on audio duration
    audio_duration = len(audio) / samplerate  # Duration in seconds
    
    # Estimate tokens needed: 
    # - Average speaking rate is ~150 words per minute
    # - Average word is ~1.5 tokens
    # - For diarization, add extra tokens for speaker labels and timestamps
    # - Add a 30% buffer for safety
    estimated_tokens = int(audio_duration * (150/60) * 1.5 * 1.3) + 200  # +200 for speaker labels/formatting
    
    # Set minimum and maximum token limits
    min_tokens = 200
    max_tokens = 4096  # Adjust based on your model's context length limits
    
    # Clamp the estimated tokens between min and max values
    max_new_tokens = max(min_tokens, min(estimated_tokens, max_tokens))
    
    print(f"Audio duration: {audio_duration:.2f} seconds, Allocated tokens: {max_new_tokens}")
    
    # Process the audio and text input
    inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')
    
    # Generate the diarized transcription
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
        )
    
    # Remove the initial prompt tokens from the generated output
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    
    # Decode the generated tokens to obtain the final response
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return response

"""Main function to process all audio files with all prompt templates."""
# Initialize results dictionary
results = {}

# Iterate over each audio file
for i, audio_path in enumerate(audio_files):
    audio_name = os.path.basename(audio_path)
    print(f"\n=== Processing Audio {i+1}/{len(audio_files)}: {audio_name} ===")
    
    results[audio_name] = {}
    
    # Process with each prompt template
    for j, template in enumerate(prompt_templates):
        print(f"\n--- Template {j+1}/{len(prompt_templates)} ---")
        print(f"Template: {template}")
        
        # Process audio with the current template
        response = process_audio_with_template(audio_path, template, processor, model)
        
        # Store the result
        results[audio_name][f"template_{j+1}"] = response
        
        # Print a preview of the response
        print(f"Response preview: {response[:200]}...")

# Save results to a file
output_dir = os.path.join(os.path.dirname(audio_dir), "diarization_results")
os.makedirs(output_dir, exist_ok=True)

for audio_name, audio_results in results.items():
    output_file = os.path.join(output_dir, f"{os.path.splitext(audio_name)[0]}_results.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"Diarization Results for {audio_name}\n")
        f.write("=" * 80 + "\n\n")
        
        for template_id, response in audio_results.items():
            template_idx = int(template_id.split('_')[1]) - 1
            
            f.write(f"Template {template_idx + 1}:\n")
            f.write(f"{prompt_templates[template_idx]}\n\n")
            f.write("Response:\n")
            f.write(f"{response}\n\n")
            f.write("-" * 80 + "\n\n")

print(f"\nAll results saved to {output_dir}")


# COMMAND ----------

