import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512,
    },
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

template = """Você é um mentor especializado em Large Language Models (LLMs).
Responda de forma clara, objetiva e didática à pergunta:

{question}

Resposta:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

def get_response(question):
    return chain.run(question)
