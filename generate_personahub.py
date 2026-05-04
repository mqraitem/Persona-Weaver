import sys
sys.path.append("./")
import json
import argparse
from models.load_models import load_models
from tqdm import tqdm
import os
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

lock = Lock()

def generate_single_persona(args, setting, persona):
    model = load_models(args, system_prompt="You are a helpful Assistant!")
    model.init_history()
    prompt = [f"Adapt the following persona: {persona} so it is from the following setting {setting}."]
    response = model.generate_text(prompt)[0]
    return response 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gens', type=int, default=100)
    parser.add_argument('--model_name', type=str, default="gpt-4o")
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    with open("data/persona_hub.jsonl", "r") as f:
        personas_data = [json.loads(line) for line in f]

    random.seed(99)
    chars = pd.read_csv("data/settings.csv")

    for row in chars.itertuples():
        name = row.name
        setting = row.prompt

        out_dir = f"data/personas/personahub/{args.model_name}/" if args.temperature == 0.7 else f"data/personas/personahub_{args.temperature}/{args.model_name}/"
        fn = f"{out_dir}/{name}.json"
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(fn):
            print(f"File {fn} already exists. Skipping...")
            continue

        print(f"Generating personas for {name} in setting '{setting}'...")

        # sample without replacement
        if args.num_gens > len(personas_data):
            raise ValueError("num_gens is larger than available personas_data")

        sampled_personas = random.sample(personas_data, args.num_gens)

        all_personas = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_single_persona, args, setting, persona) 
                       for persona in sampled_personas]
            for f in tqdm(as_completed(futures), total=args.num_gens):
                try:
                    result = f.result()
                    with lock:
                        all_personas.append(result)
                except Exception as e:
                    print(f"Error generating persona: {e}")

        with open(fn, "w") as f:
            json.dump(all_personas, f, indent=2)

if __name__ == "__main__":
    main()
