import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Modelo público
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Quantização 4bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

# Carrega tokenizer e modelo SEM token
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Função de chat simples
def chat(message, history=[]):
    inputs = tokenizer(message, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

# Interface Gradio
demo = gr.Interface(fn=chat, inputs="text", outputs="text", title="Mentor LLM", theme="soft")
demo.launch()
