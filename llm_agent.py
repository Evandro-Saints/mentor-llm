import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Usa o token direto do ambiente, SEM expor no código
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),  # Evita deixar no código
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512,
    }
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Você é um mentor especializado em Large Language Models (LLMs).
Responda de forma clara, objetiva e didática à seguinte pergunta:

{question}

Resposta:
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def get_response(question):
    return chain.run(question)
