import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Punctuation marks to analyze
PUNCTUATIONS = ["!", "?", "..."]

def count_punctuation(text):
    """
    Count punctuation occurrences in text.
    Returns a dict {punct: count}.
    """
    counts = Counter()

    # Special case for ellipsis "..."
    counts["..."] = text.count("...")
    text_no_ellipsis = text.replace("...", "")  # avoid double counting

    for punct in PUNCTUATIONS:
        if punct != "...":
            counts[punct] = text_no_ellipsis.count(punct)

    return counts, len(re.findall(r"\b\w+\b", text))  # also return word count for normalization

def collect_punctuation_stats(base_output_dir, model_name, methods, chars, convos, method_to_name):
    """
    Collect punctuation statistics for each method across all answers.
    Returns dict: {method: Counter(punct), word_count: int}
    """
    stats = {method_to_name[m]: Counter() for m in methods}
    totals = {method_to_name[m]: 0 for m in methods}  # total word counts

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

                        punct_counts, word_count = count_punctuation(answer_text)
                        stats[method_to_name[method]].update(punct_counts)
                        totals[method_to_name[method]] += word_count

    return stats, totals

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

    # Collect punctuation stats
    stats, totals = collect_punctuation_stats(base_output_dir, model_name, methods, chars, convos, method_to_name)

    # Normalize to percentage of punctuation per 100 words
    df = pd.DataFrame(stats).fillna(0)
    for m in methods:
        if totals[method_to_name[m]] > 0:
            df[method_to_name[m]] = (df[method_to_name[m]] / totals[method_to_name[m]]) * 100

    df = df.loc[PUNCTUATIONS]  # enforce order
    print("\nPunctuation usage (% per 100 words):\n", df.round(2))

    # Reshape for seaborn
    df_reset = df.reset_index().melt(id_vars="index", var_name="Method", value_name="Percentage")
    df_reset.rename(columns={"index": "Punctuation"}, inplace=True)

    # Plot with seaborn
    sns.set_theme()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_reset, x="Method", y="Percentage", hue="Punctuation", palette="Set2")

    # ax.set_title("Punctuation Usage by Method (per 100 words)", fontsize=18)
    ax.set_title("", fontsize=18)

    ax.set_ylabel("Percentage (%)", fontsize=18)
    ax.set_xlabel("Method", fontsize=18)
    # ax.tick_params(axis="x", labelsize=18)
    # ax.tick_params(axis="y", labelsize=18)
    plt.legend(title="Punctuation", title_fontsize=18, fontsize=18)
    # plt.tight_layout()

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)   
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)


    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/interaction_punctuation_{model_name}.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
