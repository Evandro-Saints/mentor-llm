import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def chat(message, history=[]):
    prompt = message
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

demo = gr.Interface(fn=chat, inputs="text", outputs="text", title="Mentor LLM", theme="soft")

demo.launch()
