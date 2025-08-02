---
title: Mentor LLM
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "4.19.2"
app_file: app.py
pinned: false
---

# Mentor LLM 🤖🧠

Este é um assistente LLM baseado no modelo [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct), rodando com interface Gradio no Hugging Face Spaces.

## Requisitos

- Transformers
- PyTorch
- Accelerate
- Gradio
- Flash Attention (se rodar em GPU)

## Observação

Esse Space foi otimizado para rodar **mesmo em CPU**. Se houver GPU disponível, será usada automaticamente com suporte a `bfloat16` e `FlashAttention-2`.

---

### 🔗 Links úteis

- Modelo base: [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
- Documentação Hugging Face: [huggingface.co/docs](https://huggingface.co/docs)
