# Breaking the Assistant Mold: Modeling Behavioral Variation in LLM Based Procedural Character Generation

### Models Setup

The pipeline uses three frontier models behind three different providers. Place each provider's API key (one key per file, no surrounding whitespace) at:

| File | Provider | Used for |
| --- | --- | --- |
| `api_keys/openai_api_key.txt` | OpenAI | GPT-4o |
| `api_keys/groq_api_key.txt` | Groq | LLaMA 3.3 70B Versatile (`llama-3.3-70b-versatile`) and Qwen 3 32B (`qwen/qwen3-32b`) |

All three files are git-ignored.

### PersonaHub Setup 

For the personahub method to work, please make sure you have the 1B personas file: persona_hub.jsonl under data/. 

You can obtain it from [here](https://huggingface.co/datasets/proj-persona/PersonaHub) 


### Generating Characters 

To generate characters, run the following:

```
python generate_[method].py --model_name [gpt-4o/llama/qwen]
```

The method can be either:

* worldweaver
* personahub
* personaweaver_moral
* personaweaver_interaction

Each script defaults to 100 characters per setting (matching the paper). Override with `--num_gens N` for `worldweaver` / `personahub`, or `--n_personas N` for the two `personaweaver_*` scripts.

### Generating Interactions 

We study two interactions in our paper: 

1- Social Norm Questions. Tagged as "moral" 

2- Interaction Style. Tagged as "interaction" 

For 1, run:

```
python get_answers_moral.py --model_name [gpt-4o/llama/qwen] --method [worldweaver/personahub/personaweaver_moral]
```

For 2, run:

```
python get_answers_interaction.py --model_name [gpt-4o/llama/qwen] --method [worldweaver/personahub/personaweaver_interaction]
```

Each answer script iterates over every persona file generated in the previous step. Both are resumable: persona files whose answer JSON already exists are skipped.

### Results

To replicate the plots in our paper, please look into the plotting files: 

```
plot_[plot type].py
```

* answered: Reaction distribution (refusal / deflection / compliance). Run as `python plot_answered.py --model_name_script [gpt-4o/llama/qwen]`. Calls Qwen 3 32B at temperature 0.1 as an auxiliary classifier.
* fillers: Filler words plot.
* length: Answer length plot.
* moral_dist: Distribution of Social Norms MCQ answers.
* punc: Punctuation plot.
* sentiment_dist: Sentiment distribution of interaction answers.


