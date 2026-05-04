import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def collect_moral_answers(base_output_dir, model_name, methods, chars, questions, method_names):
    """
    Collect moral question answers (1–4) per method across settings/questions.
    Returns DataFrame with rows=methods, cols=[1,2,3,4]
    """
    answer_counts = {method_names[m]: {1: 0, 2: 0, 3: 0, 4: 0} for m in methods}

    for row in chars.itertuples():
        setting = row.name

        for question_idx, question in enumerate(questions):
            for method in methods:
                output_dir = os.path.join(
                    base_output_dir, method, model_name, setting, f"answers_moral_{question_idx}"
                )

                if not os.path.exists(output_dir):
                    continue

                for answer_file in os.listdir(output_dir):
                    if answer_file.endswith(".json"):
                        with open(os.path.join(output_dir, answer_file), "r") as f:
                            data = json.load(f)

                            # answers are numeric strings "1","2","3","4"
                            try: 
                                ans = int(data["Dialogue"]["R1"][0])
                            except:
                                continue
                            if ans in [1, 2, 3, 4]:
                                answer_counts[method_names[method]][ans] += 1

    df = pd.DataFrame(answer_counts).T.fillna(0)
    df = df[[1, 2, 3, 4]]  # enforce order
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama")
    args = parser.parse_args()
    model_name = args.model_name

    method_names = {
        "worldweaver": "World Weaver",
        "personahub": "Personahub",
        "personaweaver_moral": "PersonaWeaver (Ours)",
    }
    
    methods = ["worldweaver", "personahub", "personaweaver_moral"]
    base_output_dir = "data/outputs/"

    chars = pd.read_csv("data/settings.csv")
    convos = pd.read_csv("data/moral_questions.csv")
    questions = convos["rot"].unique().tolist()

    # Collect answers
    df = collect_moral_answers(base_output_dir, model_name, methods, chars, questions, method_names)

    # Normalize to percentages
    df_percent = df.div(df.sum(axis=1), axis=0) * 100
    print("\nMoral question answer distribution (%):\n", df_percent.round(2))


    # melt dataframe into long format
    df_long = df_percent.reset_index().melt(
        id_vars="index", 
        var_name="Answer", 
        value_name="Percentage"
    )
    df_long.rename(columns={"index": "Method"}, inplace=True)

    # plot stacked bar chart using histplot
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(
        data=df_long,
        x="Method",
        weights="Percentage",
        hue="Answer",
        multiple="stack",
        palette="Set2",
        shrink=0.4,
        discrete=True
    )

    plt.ylabel("Percentage (%)", fontsize=18)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlabel("", fontsize=18)

    # # legend formatting
    # plt.legend(
    #     title=None,
    #     fontsize=16,
    #     labels=["Fully Disagree", "Partially Disagree", "Partially Agree", "Fully Agree"],
    #     bbox_to_anchor=(0.5, 1.26),
    #     loc="upper center",
    #     ncol=2,
    # )

    # legend formatting
    plt.legend(
        title=None,
        fontsize=16,
        labels=["Fully Disagree", "Partially Disagree", "Partially Agree", "Fully Agree"],
        bbox_to_anchor=(-0.01, 1.0),
        loc="upper left",
        ncol=2,
    )

    plt.title("Moral Answers Distribution", fontsize=18)


    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/moral_answers_{model_name}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
