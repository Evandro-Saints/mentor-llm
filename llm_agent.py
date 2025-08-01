from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json

with open("prompt_config.json") as f:
    config = json.load(f)

llm = HuggingFaceHub(
    repo_id="deepseek-ai/deepseek-llm-7b-chat",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

template = config["template"]
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

def get_response(question):
    return chain.run(question)
