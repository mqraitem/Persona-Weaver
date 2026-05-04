import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.load_models import load_models
import pandas as pd

def load_persona_cards(path):
    print(path)
    # quit()
    with open(path, 'r') as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def generate_conversation_threadsafe(persona, index, output_path, question, method, args):

    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping...")
        return f"File {output_path} already exists. Skipping..."

    while True:
        try:

            system_prompt = (
                "You are good at acting!"
                "You are given a persona and a question. "
                "You ONLY answer the question directly in character, without narration or stage directions. "
                "Do not include inner thoughts, pauses, or descriptions of actions. "
            )

            # Create model in thread
            model = load_models(
                                args, 
                                system_prompt = system_prompt, 
                                max_token=50
                                )
                                
            model.init_history()

            # System prompt
            desc_prompt = f"""
                        You are the following persona:
                        {persona}
                        """
            q1 = question
            prompt = f"{desc_prompt}\n Someone asks you: {q1} \You Say: "
            r1 = model.generate_text([prompt])[0].strip()

            print(r1)
            print("="*50)
            result = {
                "persona": persona,
                "Dialogue": { 
                    "Q1": q1,
                    "R1": r1,
                }
            }

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return f"✓ Persona saved."

        except Exception as e:
            print(f"⚠ Retrying for {persona}: {e}")
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_workers', type=int, default=10)
    args = parser.parse_args()

    threads = True

    # Paths
    base_input_dir = f"data/personas/{args.method}/{args.model_name}/"
    base_output_dir = f"data/outputs/{args.method}/{args.model_name}/"
    ensure_dir(base_output_dir)

    # setting = args.setting
    chars = pd.read_csv("data/settings.csv")
    for row in chars.itertuples():
        # name = row.name
        setting = row.name
        convos = pd.read_csv('data/open_ended_questions.csv')
        for row in convos.itertuples():        
            name = row.name 
            question = row.question

            persona_path = os.path.join(base_input_dir, f"{setting}.json")
            output_dir = os.path.join(base_output_dir, setting, f"answers_{name}")

            ensure_dir(output_dir)
            personas = load_persona_cards(persona_path)

            # Model parameters for threading            
            print(f"🧠 Generating {len(personas)} dialogues with {args.max_workers} threads...")

            if threads: 
                # Threaded generation
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    futures = []
                    for i, persona in enumerate(personas):
                        out_path = os.path.join(output_dir, f"persona_{i}.json")
                        futures.append(executor.submit(generate_conversation_threadsafe, persona, i, out_path, question, args.method, args))

                    for f in tqdm(as_completed(futures), total=len(futures), desc=f"Generating for {setting}"):
                        print(f.result())

            else: 
                # Sequential generation
                for i, persona in tqdm(enumerate(personas), total=len(personas), desc=f"Generating for {setting}"):
                    out_path = os.path.join(output_dir, f"persona_{i}.json")
                    generate_conversation_threadsafe(persona, i, out_path, question, args.method, args)

if __name__ == "__main__":
    run()
