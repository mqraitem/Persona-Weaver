import requests
import time
import base64

class QwenModel():
	def __init__(self, system_prompt, api_key, model, temperature = 0.5, max_token = 100, num_completions=1):

		self.system_prompt = system_prompt
		self.history = None
		
		self.history = self.init_history()
		self.headers = {
			"Content-Type": "application/json",
			"Authorization": f"Bearer {api_key}"
		}

		self.max_token = max_token 
		self.model = model
		self.temperature = temperature
		self.num_completions = num_completions

	def init_history(self):
		self.history = {}
		self.n_runs = 0 
	
	def set_history(self, history):
		self.history = history
	
	def get_content_message(self, content): 
		
		return  [
			{
				"type": "text",
				"text": content
				}
		]

	def first_call(self, prompt):
		
		payload = {
			"model": "qwen/qwen3-32b",
			"messages": 
			[
				{
					"role": "system",
					"content": self.system_prompt
				},
				{
					"role": "user",
					"content": self.get_content_message(prompt),
					 
				},
			],
			"max_tokens": self.max_token,
			"temperature": self.temperature,
			"reasoning_effort": "none"
		}

		return payload

	def update_history(self, response):

		response_message = {"role": "assistant", 
							"content":  response
		}

		self.history["messages"] += [response_message]
		self.n_runs += 1

	def subsequent_call(self, prompt):
		prompt = {"role": "user",
			"content": self.get_content_message(prompt)
		}
		self.history["messages"] += [prompt]
		return self.history


	def make_api_call(self, payload):

		while True:
			try:
				response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=self.headers, json=payload)
				response = response.json()
				response_content = [response['choices'][i]['message']['content'] for i in range(self.num_completions)]
				if 'API limit' in response_content[0].lower():
					print("API limit reached, sleeping for 10 seconds...")
					time.sleep(10)
					continue

				return response_content

			except Exception as e:
				print(f"Encountered an error: {e}, sleeping for 10 seconds...")
				time.sleep(10)


	def generate_text(self, prompt):

		assert len(prompt) == 1, "Number of prompts must be 1"
		prompt = prompt[0]
		if self.n_runs == 0:
			payload = self.first_call(prompt)
			self.history = payload
		else:
			payload = self.subsequent_call(prompt)
		response = self.make_api_call(payload)

		self.update_history(response[0])

		for idx, r in enumerate(response):
			if "</think>" in r:
				response[idx] = r.split("</think>")[1] 
		return response