import os
import json
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import sys
import time
import random
import itertools
import concurrent.futures
sys.path.append("./")
from models.load_models import load_models

def generate_combinations(attributes):
    keys = list(attributes.keys())
    values = list(attributes.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]

def sample_combinations(attributes, n):
    """
    Efficiently sample n random combinations of attributes 
    without generating the full Cartesian product.
    
    Parameters:
    - attributes: dict of {attribute_name: list_of_values}
    - n: number of combinations to sample

    Returns:
    - list of dicts, each a sampled attribute combination
    """
    keys = list(attributes.keys())
    combos = []
    for _ in range(n):
        combo = {k: random.choice(attributes[k]) for k in keys}
        combos.append(combo)
    return combos

def load_moral_positions(path="data/moral_positions.csv"):
    df = pd.read_csv(path)
    return {"Your guidance for making moral decisions": df["position"].tolist()}


moral_positions = load_moral_positions()
moral_motives_list = generate_combinations(moral_positions)
NUM_CATEGORIES = 10

def generate_backstory(args, persona: Dict[str, any], setting: str) -> Dict[str, str]:
    model = load_models(args)
    while True:
        try:
            model.init_history()
            retries = random.randint(1, 4)
            for i in range(retries):
                if i == 0:
                    prompt = f"""
                            You are generating the **backstory** for a fictional character.

                            Setting: {setting}
                            Card:
                            {json.dumps(persona, indent=2)}

                            Describe the character's **backstory** — . (at most 50  words)

                            Respond in JSON format:
                            {{
                            "backstory": "..."
                            }}
                    """
                else:
                    prompt = """
                        Retry while increasing the emotional contrast and drama!

                        Respond in JSON format:
                        {{
                        "backstory": "..."
                        }}
                        (at most 30 words)
                    """

                response = model.generate_text([prompt])[0].strip()
                json_start = response.index('{')
                json_end = response.rindex('}') + 1
                parsed = json.loads(response[json_start:json_end])
                persona.update(parsed)
                return persona

        except Exception as e:
            print(f"⚠️ Retry after failure: {e}")
            continue
            

def fix_inconsistent_attributes(args, persona_card, setting) -> Dict[str, str]:
    """
    Pass the sampled attributes to the model and ask it to adjust or rewrite
    them for internal consistency, coherence, and setting-appropriateness.
    """
    prompt = f"""
    You are checking sampled character card for consistency.

    Setting: {setting}
    Card:
    {json.dumps(persona_card, indent=2)}

    Task:
    - Ensure the attributes are coherent and do not contradict each other.
    - If needed, adjust wordings to harmonize them (without losing specificity).
    - Keep the same categories.
    - Do not invent new categories.

    Reproduce the Fixed Card
    """

    while True:
        try:
            model = load_models(args)
            model.init_history()
            response = model.generate_text([prompt])[0].strip()
            start = response.index('{')
            end = response.rindex('}') + 1
            return json.loads(response[start:end])
        except Exception as e:
            print(f"⚠️ Attribute fix failed, retrying: {e}")
            time.sleep(1)
            continue


def generate_attribute_categories(args, setting: str) -> List[str]:
    model = load_models(args)
    model.init_history()
    prompt = f"""
    You are helping create characters for the setting: {setting}.
    
    We need a list of only {NUM_CATEGORIES} **non-behavioral** attribute categories to describe characters in this setting (e.g., occupation, affiliation, or expertise).
    These categories must:
    - Be mutually exclusive
    - Exclude behavioral traits such as personality, moral stances, or interaction styles

    Return a JSON array like:
    ["Category1", "Category2", ...]
    """


    while True:
        response = model.generate_text([prompt])[0].strip()
        try:
            start = response.index('[')
            end = response.rindex(']') + 1
            return json.loads(response[start:end])
        except Exception as e:
            print(f"Error parsing categories: {e}")
            time.sleep(1)
            continue


def generate_attributes_batch(args, category: str, setting: str, total_attr: int = 20) -> List[Dict[str, str]]:
    model = load_models(args)
    model.init_history()
    prompt = f"""
    You are helping define non-behavioral, vivid attributes for a character in the setting: "{setting}".
    Attribute Category: "{category}"

    Generate {total_attr} possibilities of {category}. Each possibility should hollistic of the category. 

    Return a JSON array like:
    [
      {{
        "option": 
        "description": 
      }},
      ...
    ]

    description is at most 20 words.
    """

    while True:
        response = model.generate_text([prompt])[0].strip()
        try:
            start = response.index('[')
            end = response.rindex(']') + 1
            batch = json.loads(response[start:end])
            if isinstance(batch, list) and all('option' in item and 'description' in item for item in batch):
                return batch[:total_attr]
        except Exception as e:
            print(f"Error parsing batch attributes for category '{category}': {e}")
            time.sleep(1)
            continue

def process_persona(args, persona_moral_motives, persona_attributes, setting):

    persona_card = {
        "You Attributes": persona_attributes,
        "Your moral guidance (Please follow these instructions as closely as possible)": persona_moral_motives,
    }

    persona_card = fix_inconsistent_attributes(args, persona_card, setting)
    return persona_card


def generate_persona_cards(args, n: int, setting: str, attributes: Dict[str, List[Dict[str, str]]]):
    cards = []

    selected_moral_motives = random.choices(moral_motives_list, k=n)
    selected_attributes =  sample_combinations(attributes, n)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for persona_moral_motives, persona_attributes in zip(selected_moral_motives, selected_attributes):
            futures.append(executor.submit(process_persona, args, persona_moral_motives, persona_attributes, setting))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"processing"):
            cards.append(future.result())

    return cards

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gpt-4o")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--n_personas', type=int, default=100)
    args = parser.parse_args()

    chars = pd.read_csv("data/settings.csv")
    out_dir = f"data/personas/personaweaver_moral/{args.model_name}" if args.temperature == 0.7 else f"data/personaweaver_moral_{args.temperature}/{args.model_name}/"
    os.makedirs(out_dir, exist_ok=True)

    for row in chars.itertuples():
        name = row.name
        setting = row.prompt

        filename = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(filename):
            continue

        print(f"\n=== Generating personas for '{name}' in setting: {setting} ===")
        filename_attrs = os.path.join(out_dir, f"{name}_attributes.json")

        if os.path.exists(filename_attrs):
            with open(filename_attrs, 'r') as f:
                attributes = json.load(f)
        else: 
            # Generate real attributes
            attributes = {}
            categories = generate_attribute_categories(args, setting)
            print(f"Generated categories: {categories}")
            for category in categories:
                attributes[category] = generate_attributes_batch(args, category, setting, total_attr=30)
                for attr in attributes[category]:
                    print(f"Category: {category}, Attribute: {attr['option']}, Description: {attr['description']}")
                    print("="*10)

            with open(os.path.join(out_dir, f"{name}_attributes.json"), 'w') as f:
                json.dump(attributes, f, indent=2)

        persona_cards = generate_persona_cards(args, args.n_personas, setting, attributes)
        with open(filename, 'w') as f:
            json.dump(persona_cards, f, indent=2)


if __name__ == "__main__":
    main()


