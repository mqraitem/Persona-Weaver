
import sys
sys.path.append("./")
import argparse
from models.load_models import load_models
import pandas as pd
import os 

def main(): 

    parser=argparse.ArgumentParser()
    parser.add_argument('--num_gens', type=int, default=100)
    parser.add_argument('--model_name', type=str, default="gpt-4o")
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()

    system_prompt = "You are a helpful Assistant!"
    model = load_models(args, system_prompt, max_token=8000)


    chars = pd.read_csv("data/settings.csv")
    for row in chars.itertuples():
        name = row.name
        setting = row.prompt

        print(f"Generating personas for {setting}...")

        all_personas = []        
        to_generate = 50

        out_dir = f"data/personas/worldweaver/{args.model_name}/" if args.temperature == 0.7 else f"data/personas/worldweaver_{args.temperature}/{args.model_name}/"
        fn = f"{out_dir}/{name}.json"

        if os.path.exists(fn):
            print(f"File {fn} already exists. Skipping...")
            continue

        while len(all_personas) < args.num_gens:

            while True: 
                try: 
                    model.init_history()

                    prompt = f"Generate {to_generate} different personas/characters cards in the following setting {setting}."
                    prompt += "You should always populate name, description, persona, goal, and properties of each character."
                    prompt += "The output should be a python list. Don't include python comments in the list! Ensure the output is a valid python list and could be parsed by ast"
                    
                    responses = model.generate_text([prompt])
                    first_index_list = responses[0].index("[")
                    last_index_list = responses[0].rindex("]") + 1

                    response = responses[0][first_index_list:last_index_list]
                    response = response.replace("\n", "")
                    print(f"Generated persona: {response}")
                    print("="*100)

                    # print(response)
                    # quit()
                    #parse the list 
                    import ast
                    response = ast.literal_eval(response)
                    all_personas.extend(response)
                    break 
                except Exception as e:
                    print(f"Error generating persona: {e}, retrying...")
                    continue

        all_personas = all_personas[:args.num_gens]
        os.makedirs(out_dir, exist_ok=True)
        import json
        with open(fn, "w") as f:
            json.dump(all_personas, f, indent=2)


def get_gen(prompt, model):

    response = model.generate_text(prompt)
    return response


if __name__ == "__main__":
    main()