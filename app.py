import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Verifica se há GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega o modelo com fallback para CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
).to(device)

def chat(message, history=None):
    inputs = tokenizer(message, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

demo = gr.Interface(fn=chat, inputs="text", outputs="text", title="Mentor LLM")

demo.launch()
