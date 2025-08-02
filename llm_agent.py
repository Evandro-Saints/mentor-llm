from huggingface_hub import InferenceClient
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

import os

# Substitua pelo seu token Hugging Face (ou carregue via dotenv)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HUGGINGFACE_TOKEN
)

def get_response(question):
    # Gera texto com parâmetros definidos
    response = client.text_generation(
        prompt=question,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1
    )
    return response.strip()
