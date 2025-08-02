import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

# ⚠️ Pegue o token via variável de ambiente
hf_token = os.getenv("HF_TOKEN")

# Prompt que o modelo vai seguir
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Você é um mentor especializado em Large Language Models (LLMs).
Responda de forma clara, objetiva e didática à seguinte pergunta:

{question}

Resposta:
"""
)

# Instanciando o modelo
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",  # ou outro que preferir
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# Ligando o prompt ao modelo (forma nova com RunnableSequence)
chain = prompt | llm

def get_response(question):
    return chain.invoke({"question": question})
