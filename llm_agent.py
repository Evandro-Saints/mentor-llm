import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Instancia o modelo
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512
    }
)

# Prompt base
prompt = PromptTemplate(
    input_variables=["question"],
    template="Responda de forma clara e objetiva a pergunta: {question}"
)

# Cadeia de execução
chain = LLMChain(llm=llm, prompt=prompt)

def get_response(question):
    return chain.run(question)
