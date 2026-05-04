import os
import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme() 

from concurrent.futures import ThreadPoolExecutor, as_completed

from models.load_models import load_models
from tqdm import tqdm

def classify_answer(question, answer, args):
    """
    Ask the model to classify the answer into one of three numeric categories:
    1 = Reject to answer
    2 = Provided an answer but deflected
    3 = Answered the question
    """
    prompt = f"""
You are evaluating dialogue quality.

Question: "{question}"
Answer: "{answer}"

Classify the answer into exactly one of these categories:

1 = Reject to answer → The character rekects to provide an answer. 
2 = Deflected → The character responds, but doesn't answer the question. 
3 = Answered the question → The character answers the question.

Respond ONLY with the number 1, 2, or 3.
"""

    model = load_models(args)  # load model per call
    response = model.generate_text([prompt])[0].strip()
    # Extract the first digit if present
    if response.startswith("1"):
        return "1"
    elif response.startswith("2"):
        return "2"
    else:
        return "3"


def process_answer_file(method, question_text, filepath, args):
    """Helper to process a single answer file."""
    with open(filepath, "r") as f:
        data = json.load(f)
        answer_text = data["Dialogue"]["R1"]

    result = classify_answer(question_text, answer_text, args)
    return method, result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_script", type=str, default="llama")
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_workers", type=int, default=2)
    args = parser.parse_args()
    model_name = args.model_name_script

    methods = ["worldweaver", "personahub", "personaweaver_interaction"]
    base_output_dir = "data/outputs/"

    method_to_name = {
        "worldweaver": "World Weaver",
        "personahub": "Personahub",
        "personaweaver_interaction": "Personaweaver (Ours)"
    }

    chars = pd.read_csv("data/settings.csv")
    convos = pd.read_csv("data/open_ended_questions.csv")

    os.makedirs("plots/answered_question", exist_ok=True)

    # Store total counts across all runs
    global_counts = {method_to_name[m]: Counter() for m in methods}

    categories = ["1", "2", "3"]
    cat_to_label = {
        "1": "Reject to answer",
        "2": "Answered but Deflect",
        "3": "Answered the question",
    }

    for row in chars.itertuples():
        setting_name = row.name

        for row_ in convos.itertuples():
            question_name = row_.name
            question_text = row_.question

            out_csv = f"plots/answered_question/{setting_name}_{question_name}_{model_name}.csv"
            if os.path.exists(out_csv):
                print(f"✅ Skipping {setting_name}/{question_name} (already processed)")
                df_existing = pd.read_csv(out_csv, index_col=0)
                for method, rowvals in df_existing.iterrows():
                    for cat in categories:
                        global_counts[method][cat] += rowvals.get(cat, 0)
                continue

            print(f"\n--- Processing {setting_name}/{question_name} ---")

            # Per (setting, question) counts
            local_counts = {method_to_name[m]: Counter() for m in methods}

            for method in methods:
                method_path = os.path.join(
                    base_output_dir, method, model_name, setting_name, f"answers_{question_name}"
                )
                if not os.path.exists(method_path):
                    continue

                futures = []
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    for answer_file in os.listdir(method_path):
                        if answer_file.endswith(".json"):
                            filepath = os.path.join(method_path, answer_file)
                            futures.append(
                                executor.submit(
                                    process_answer_file,
                                    method_to_name[method],
                                    question_text,
                                    filepath,
                                    args,
                                )
                            )

                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Processing {method} ({setting_name}/{question_name})",
                    ):
                        method_, result = future.result()
                        local_counts[method_].update([result])
                        global_counts[method_].update([result])

                print(local_counts)

            # Save per-question results (numbers only)
            df_local = pd.DataFrame.from_dict(local_counts, orient="index").fillna(0)
            df_local = df_local.reindex(columns=categories, fill_value=0)
            df_local.to_csv(out_csv)
            print(f"💾 Saved results for {setting_name}/{question_name} → {out_csv}")
   

    # --- Final aggregation ---
    df = pd.DataFrame.from_dict(global_counts, orient="index").fillna(0)
    df = df.reindex(columns=categories, fill_value=0)

    # convert to percentages
    df_percent = df.div(df.sum(axis=1), axis=0) * 100
    print("\nGlobal Answer Evaluation (%):\n", df_percent.round(2))

    # Reshape for seaborn (map back to labels for readability)
    df_reset = df_percent.reset_index().melt(
        id_vars="index", var_name="Category", value_name="Percentage"
    )
    df_reset.rename(columns={"index": "Method"}, inplace=True)

    # Apply label mapping safely (avoids NaN if dictionary is incomplete)
    df_reset["Category"] = df_reset["Category"].replace(cat_to_label).fillna(df_reset["Category"])

    # --- Plot with seaborn histplot ---
    # sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    order = ["Answered the question", "Answered but Deflect", "Reject to answer"]
    df_reset["Category"] = pd.Categorical(df_reset["Category"], categories=order, ordered=True)

    ax = sns.histplot(
        data=df_reset,
        x="Method",
        weights="Percentage",
        hue="Category",
        multiple="stack",
        palette="Set2",
        shrink=0.4,   # controls bar width
        discrete=True
    )

    # --- Formatting ---
    ax.set_title("Did the Characters Answer the Question?", fontsize=18)
    ax.set_ylabel("Percentage (%)", fontsize=18)
    ax.set_xlabel("", fontsize=18)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    plt.legend(
        title=None,
        labels=["Reject to answer", "Answered but Deflect", "Answered the question"],
        fontsize=16,
        bbox_to_anchor=(-0.01, 1.0),
        loc="upper left",
        ncol=1,
    )

    plt.tight_layout()

    # save outputs
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/answered_question_summary_{model_name}.png", dpi=300)
    df.to_csv(f"plots/answered_question_summary_{model_name}.csv")

    plt.show()

if __name__ == "__main__":
    main()
