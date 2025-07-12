from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Log in to Hugging Face using the token from .env
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
class Generator:
    def __init__(self, model_name='HuggingFaceH4/zephyr-7b-beta', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto'
        )
        self.model.eval()

    def build_prompt(self, context_chunks, user_query):
        context = "\n\n".join(context_chunks)
        prompt = (
            f"<|system|>\nYou are an AI assistant answering questions strictly based on provided context.\n\n"
            f"<|user|>\nContext:\n{context}\n\nQuestion: {user_query}\n\n<|assistant|>\n"
        )
        return prompt

    def generate_stream(self, prompt, max_new_tokens=512, temperature=0.2):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            inputs=inputs['input_ids'],
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token