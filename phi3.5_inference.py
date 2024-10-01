import torch
import gc
import pandas as pd
import requests
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
import os
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define constants
OUTPUT_FILE = "phi3-5_xkcd_results.tsv"
ERROR_LOG_FILE = "error_log.txt"

# Load the xkcd dataset from Hugging Face
dataset = load_dataset("olivierdehaene/xkcd")

def get_image_from_filename(filename):
    """Load image from local file."""
    try:
        image_path = os.path.join('xkcd_images', filename)
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {filename}: {e}")
        return None


question = '''Analyze the comic and explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the comic and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any). Answer in a single paragraph.'''

model_name = "microsoft/Phi-3.5-vision-instruct"

# Initialize LLM with tensor_parallel_size=2 to run on 2 GPUs
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_num_seqs=1,
    gpu_memory_utilization=0.4,  # Adjust this value based on your setup
    max_model_len=4000,  # Decrease if necessary
    # device='cuda:1'  # Set device to cuda:1 explicitly
    # dtype="half",  # Use half precision (float16)
    # quantization="fp8"  # Use 4-bit quantization
)


# Open the output file and error log in write mode
with open(OUTPUT_FILE, "w") as file, open(ERROR_LOG_FILE, "w") as error_log:
    # Write the header
    file.write("index\timage_url\tresponse\n")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        max_tokens=512,
    )
    
    # Process dataset one by one
    for item in tqdm(dataset['train'], desc="Processing items"):
        try:
            prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
            image = get_image_from_filename(str(item['id'])+".jpg")
            
            # input_data = run_phi3([prompt], [image_base64])
            # tokenized_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            input_data = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
            }

            output = llm.generate(input_data, sampling_params=sampling_params)
            
            generated_text = output[0].outputs[0].text
            file.write(f"{item['id']}\t{item['image_url']}\t{generated_text.strip()}\n")
            file.flush()
        
        except Exception as e:
            error_message = f"Error processing item {item['id']}: {str(e)}"
            print(error_message)
            error_log.write(f"{item['id']}\t{item['image_url']}\t{error_message}\n")
            error_log.flush()
        
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()

print(f"Results saved to {OUTPUT_FILE}")
print(f"Errors logged to {ERROR_LOG_FILE}")


