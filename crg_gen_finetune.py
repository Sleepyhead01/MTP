import json
from tqdm.notebook import tqdm
from openai import OpenAI

client = OpenAI(api_key="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")
INPUT_JSONL = "gpt-4o_causal_reasoning_graph_results.jsonl"
MODEL = "gpt-4o"
OUTPUT_JSONL = f"{MODEL}_crg_generation.jsonl"

DEBUG_MODE = False  # Toggle this to enable/disable debug mode

def process_entries():
    # Read existing JSONL file
    with open(INPUT_JSONL, 'r') as f:
        entries = [json.loads(line) for line in f]

    if DEBUG_MODE:
        entries = entries[:1]  # Process only first entry in debug mode
        print("=== DEBUG MODE ENABLED ===")

    # Create output file if not in debug mode
    if not DEBUG_MODE:
        with open(OUTPUT_JSONL, 'w') as f: pass

    # Process each entry with progress bar
    for entry in tqdm(entries, desc="Generating final responses"):
        try:
            # Format prompt with previous response
            prompt_text = f"""## Causal reasoning graph\n{entry['response']}\n ## Question\n"Analyze the image and explain the humor (if any) along with the underlying message/concept it represents. In other words, analyze the image and explain the cultural, scientific, or technical references and how they contribute to the joke or punchline (if any)." To answer this, use the causal reasoning graph above that links different objects, people, and entities present in the image."""
            url = entry['image_url']
            if DEBUG_MODE:
                print("\n=== API REQUEST ===")
                print("url: ", url)
                print(json.dumps({"prompt": prompt_text}, indent=2))

            # API call with formatted prompt
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

            content = response.choices[0].message.content

            if DEBUG_MODE:
                print("\n=== API RESPONSE ===")
                print(json.dumps({"response": content}, indent=2))
                return  # Exit after first iteration in debug mode

        except Exception as e:
            content = f"Error: {str(e)}"
            if DEBUG_MODE:
                print("\n=== ERROR ===")
                print(content)
                return

        # Append result immediately (only in normal mode)
        if not DEBUG_MODE:
            with open(OUTPUT_JSONL, 'a') as f:
                json.dump({
                    "index": entry['index'],
                    "image_url": entry['image_url'],
                    "initial_response": entry['response'],
                    "final_analysis": content
                }, f)
                f.write('\n')

process_entries()
