# !pip install -qqq openai datasets
import random
from datasets import load_dataset
import json
from tqdm.notebook import tqdm
from openai import OpenAI

# Initialize API client
client = OpenAI(api_key="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") 
MODEL = "gpt-4o"
OUTPUT_FILE = f"{MODEL}_causal_reasoning_graph_results.jsonl"

# Load dataset with split specification
dataset = load_dataset("olivierdehaene/xkcd")

# Load already processed indices
processed = set()
# try:
#     with open(OUTPUT_FILE) as f:
#         for line in f:
#             processed.add(json.loads(line)['index'])
# except FileNotFoundError:
#     pass

for index, data in tqdm(enumerate(dataset['train']), total=len(dataset['train']), desc="Processing dataset"):
    if index in processed:  # Skip excluded indices
        continue
    
    url = data['image_url']
    try:
        prompt_text = """"Analyze the image and explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the image and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any)." To answer this, create a causal reasoning graph linking different objects, people, and entities present in the image in the form of Python code. Note that you have to only create a causal graph for this step. Make sure that the final code is followed after 'FinalCode:' """  # Your prompt

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": url}}
                ]
            }],
            temperature=0.0,
        )
        content = response.choices[0].message.content if response.choices else "No response"
    except Exception as e:
        content = f"Error: {str(e)}"

    # Immediately write result to JSONL
    with open(OUTPUT_FILE, 'a') as f:
        json.dump({
            "index": index,
            "image_url": url,
            "response": content
        }, f)
        f.write('\n')
