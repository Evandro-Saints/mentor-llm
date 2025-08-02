import streamlit as st
from llm_agent import get_response

st.set_page_config(page_title="Mentor LLM")
st.title("🤖 Mentor LLM")
st.markdown("Seu orientador pessoal sobre LLMs.")

query = st.text_input("O que você quer saber sobre LLMs?")

if query:
    try:
        answer = get_response(query)
        st.success(answer)
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
