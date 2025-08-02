import streamlit as st
from llm_agent import get_response

st.set_page_config(page_title="Mentor LLM", layout="centered")
st.title("🤖 Mentor LLM")
st.markdown("Seu orientador pessoal sobre Large Language Models, direto do futuro.")
st.markdown("❓ Pergunte qualquer coisa sobre LLMs")

query = st.text_input("Digite sua pergunta aqui:")

if query:
    try:
        response = get_response(query)
        st.success(response)
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
