import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Agora com os parâmetros certos!
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=512
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="Responda de forma clara e objetiva a pergunta: {question}"
)

chain = LLMChain(llm=llm, prompt=prompt)

def get_response(question):
    return chain.run(question)
