import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"  # troca por outro público se quiser

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def chat(message):
    inputs = tokenizer(message, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

demo = gr.Interface(fn=chat, inputs="text", outputs="text", title="Mentor LLM - Versão Leve")
demo.launch()
