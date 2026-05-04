import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def count_words_chars(text):
    """Return word count and character count of a string."""
    words = re.findall(r"\b\w+\b", text)
    return len(words), len(text)

def collect_lengths(base_output_dir, model_name, methods, chars, convos, method_to_name):
    """
    Collect answer lengths (words, chars) for each method.
    Returns DataFrame with columns: method, word_count, char_count
    """
    records = []

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

                        wc, cc = count_words_chars(answer_text)
                        records.append({"method": method_to_name[method], "word_count": wc, "char_count": cc})

    return pd.DataFrame(records)

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

    df = collect_lengths(base_output_dir, model_name, methods, chars, convos, method_to_name)
    print("\nLength stats:\n", df.groupby("method")[["word_count", "char_count"]].describe())

    os.makedirs("plots", exist_ok=True)

    # --- Boxplot of word counts ---
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x="method", y="word_count")

    ax.set_title("Distribution of Answer Lengths (Word Count)", fontsize=16)
    ax.set_ylabel("Word Count", fontsize=16)
    ax.set_xlabel("Method", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

    plt.savefig(f"plots/interaction_length_boxplot_{model_name}.png")

    # --- Histogram / KDE overlay ---
    plt.figure(figsize=(8, 5))
    for method in methods:
        sns.kdeplot(
            df[df["method"] == method_to_name[method]]["word_count"], 
            label=method_to_name[method],  # show proper display name
            fill=True, 
            common_norm=False, 
            alpha=0.4
        )

    ax = plt.gca()  # grab current axis
    # ax.set_title("Length Distribution (Words per Answer)", fontsize=16)
    ax.set_title("", fontsize=18)
    ax.set_xlabel("Word Count", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    ax.legend(title="Method", fontsize=18, title_fontsize=18)

    plt.savefig(f"plots/interaction_length_distribution_{model_name}.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
