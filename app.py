import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess

# Instala flash-attn dinamicamente se houver GPU
if torch.cuda.is_available():
    try:
        import flash_attn
    except ImportError:
        subprocess.run(["pip", "install", "flash-attn"], check=True)

# Configuração do modelo
model_name = "tiiuae/falcon-7b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    attn_implementation="flash_attention_2" if device == "cuda" else None
).to(device)

# Função de chat simples
def chat(message, history=None):
    inputs = tokenizer(message, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interface Gradio
demo = gr.Interface(fn=chat, inputs="text", outputs="text", title="Mentor LLM")
demo.launch()
