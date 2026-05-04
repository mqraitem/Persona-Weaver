from models.llama_model import LLaMaModel
from models.qwen_model import QwenModel
from models.gpt4_model import GPT4Model

def load_models(args, system_prompt="You are a helpful Assistant!", max_token=3000):

    if args.model_name == "llama":
        args.batch_size = 1
        api_key = open("api_keys/groq_api_key.txt", "r").read().strip()
        model = LLaMaModel(system_prompt, api_key, args.model_name, temperature = args.temperature, max_token = max_token, num_completions=args.batch_size)

    elif args.model_name == "qwen":
        args.batch_size = 1
        api_key = open("api_keys/groq_api_key.txt", "r").read().strip()
        model = QwenModel(system_prompt, api_key, args.model_name, temperature = args.temperature, max_token = max_token, num_completions=args.batch_size)

    elif args.model_name == "gpt-4o":
        args.batch_size = 1
        api_key = open("api_keys/openai_api_key.txt", "r").read().strip()
        model = GPT4Model(system_prompt, api_key, args.model_name, temperature = args.temperature, max_token = max_token, num_completions=args.batch_size)

    else:
        raise ValueError("Model name not recognized")

    return model
