import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "tiiuae/falcon-7b-instruct"

# Configuração para rodar em GPU com quantização 4bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

def chat(message, history=[]):
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

demo = gr.Interface(fn=chat, inputs="text", outputs="text", title="Mentor LLM", theme="soft")
demo.launch()
