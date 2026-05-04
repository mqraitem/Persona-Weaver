import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Common filler words/phrases (extendable)
FILLER_WORDS = {
    "um", "uh", "like", "you know", "i mean", "actually", "basically",
    "so", "well", "right", "ok", "okay", "hmm", "err", "ah", "oh"
}

def count_fillers(text):
    """
    Count filler words in a given text.
    Returns (filler_count, total_word_count).
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    total_words = len(tokens)

    filler_count = 0
    for filler in FILLER_WORDS:
        filler_count += len(re.findall(r"\b" + re.escape(filler) + r"\b", text))

    return filler_count, total_words

def collect_filler_stats(base_output_dir, model_name, methods, chars, convos, method_to_name):
    """
    Collect filler usage statistics for each method.
    Returns dict: {pretty_method_name: {"filler": int, "total": int}}
    """
    stats = {method_to_name[m]: {"filler": 0, "total": 0} for m in methods}

    for row in chars.itertuples():
        setting_name = row.name

        for row_ in convos.itertuples():
            question_name = row_.name

            for method in methods:
                method_path = os.path.join(
                    base_output_dir, method, model_name, setting_name, f"answers_{question_name}"
                )

                if not os.path.exists(method_path):
                    continue

                for answer_file in os.listdir(method_path):
                    if answer_file.endswith(".json"):
                        with open(os.path.join(method_path, answer_file), "r") as f:
                            data = json.load(f)
                            answer_text = data["Dialogue"]["R1"]

                        filler_count, total_count = count_fillers(answer_text)
                        stats[method_to_name[method]]["filler"] += filler_count
                        stats[method_to_name[method]]["total"] += total_count

    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    args = parser.parse_args()
    model_name = args.model_name

    methods = ["worldweaver", "personahub", "personaweaver_interaction"]
    base_output_dir = "data/outputs/"

    method_to_name = {
        "worldweaver": "World Weaver",
        "personahub": "Personahub",
        "personaweaver_interaction": "Personaweaver (Ours)"
    }

    chars = pd.read_csv("data/settings.csv")
    convos = pd.read_csv("data/open_ended_questions.csv")

    # Collect filler stats
    stats = collect_filler_stats(base_output_dir, model_name, methods, chars, convos, method_to_name)

    # Convert to percentages
    percentages = {
        m: (100 * stats[m]["filler"] / stats[m]["total"]) if stats[m]["total"] > 0 else 0
        for m in stats
    }

    df = pd.DataFrame.from_dict(percentages, orient="index", columns=["Filler %"]).reset_index()
    df.rename(columns={"index": "Method"}, inplace=True)

    print("\nFiller word usage (%):\n", df.round(2))

    # --- Plot as a single stacked bar ---
    plt.figure(figsize=(3, 5))

    colors = sns.color_palette("Set2", n_colors=len(df))

    bottom_val = 0
    for i, row in df.iterrows():
        plt.bar(
            x=["Methods"],
            height=row["Filler %"],
            bottom=bottom_val,
            label=row["Method"],
            color=colors[i],
        )
        bottom_val += row["Filler %"]

    # plt.title("Filler Word Usage by Method", fontsize=16)
    plt.ylabel("Percentage of Words (%)", fontsize=18)
    plt.xlabel("")
    plt.xticks([])  # Hide x-axis labels since it's just one bar
    plt.yticks(fontsize=18)

    plt.legend(title="Method", fontsize=18, title_fontsize=18,  bbox_to_anchor=(0.5, 1.45), loc="upper center", ncol=1)
    # plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/interaction_filler_usage_stacked_{model_name}.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
