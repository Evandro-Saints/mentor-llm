import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Recupera o token dos secrets da Hugging Face Space
token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=token,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
)

template = """Você é um mentor especializado em Large Language Models (LLMs).
Responda de forma clara, objetiva e didática à pergunta:

{question}

Resposta:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

chain = LLMChain(llm=llm, prompt=prompt)

def get_response(question):
    return chain.run(question)
