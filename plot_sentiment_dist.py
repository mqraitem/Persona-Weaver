import os
import json
import pandas as pd
from collections import Counter
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load sentiment model (3 classes: NEG, NEU, POS)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# The 3 sentiment labels (matches model id2label order: 0=neg, 1=neu, 2=pos)
SENTIMENTS = ["NEG", "NEU", "POS"]

def predict_sentiments(texts):
    """Return predicted sentiment labels for a list of texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    return [SENTIMENTS[p.item()] for p in preds]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    args = parser.parse_args()
    model_name = args.model_name

    methods = ["worldweaver", "personahub", "personaweaver_interaction"]
    base_output_dir = "data/outputs/"

    chars = pd.read_csv("data/settings.csv")
    convos = pd.read_csv("data/open_ended_questions.csv")

    method_to_name = { 
        "worldweaver": "World Weaver",
        "personahub": "Personahub",
        "personaweaver_interaction": "Personaweaver (Ours)"
    }

    # Store total distributions per method
    sentiment_counts_by_method = {method_to_name[m]: Counter() for m in methods}

    for row in chars.itertuples():
        setting_name = row.name

        for row_ in convos.itertuples():
            question_name = row_.name

            print(f"\n--- Setting: {setting_name}, Question: {question_name} ---")

            for method in methods:
                method_path = os.path.join(
                    base_output_dir, method, model_name, setting_name, f"answers_{question_name}"
                )

                if not os.path.exists(method_path):
                    continue

                answers = []
                for answer in os.listdir(method_path):
                    if answer.endswith(".json"):
                        with open(os.path.join(method_path, answer), "r") as f:
                            data = json.load(f)
                            answers.append(data["Dialogue"]["R1"])

                if answers:
                    predicted = predict_sentiments(answers)
                    sentiment_counts_by_method[method_to_name[method]].update(predicted)

    # Convert counts into a DataFrame
    df = pd.DataFrame.from_dict(sentiment_counts_by_method, orient="index").fillna(0)
    df = df[SENTIMENTS]  # enforce order

    # Normalize to percentages
    df_percent = df.div(df.sum(axis=1), axis=0) * 100
    print("\nSentiment distribution (%):\n", df_percent.round(2))

    # Plot with seaborn
    sns.set_theme()  # use seaborn default theme
    df_percent_reset = df_percent.reset_index().melt(id_vars="index", value_vars=SENTIMENTS,
                                                     var_name="Sentiment", value_name="Percentage")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_percent_reset, x="index", y="Percentage", hue="Sentiment", palette="Set2")
    # ax.set_title("Sentiment Distribution by Method", fontsize=16)
    ax.set_ylabel("Percentage (%)", fontsize=18)
    ax.set_xlabel("Method", fontsize=18)
    plt.legend(title="Sentiment", fontsize=18, title_fontsize=18)
    ax.set_title("", fontsize=18)
    #increase font of title in legend

    # plt.tight_layout()

    #increase font of x and y labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)   
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    #incrase font of legend 

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/interaction_sentiment_dist_{model_name}.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
