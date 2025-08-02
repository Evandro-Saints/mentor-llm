import streamlit as st
from llm_agent import get_response

st.set_page_config(page_title="Mentor LLM", page_icon="🤖")
st.title("🤖 Mentor LLM")
st.markdown("Seu orientador pessoal sobre Large Language Models, direto do futuro.")

query = st.text_input("❓ Pergunte qualquer coisa sobre LLMs", "")

if st.button("Enviar") or query:
    with st.spinner("Consultando o Mentor..."):
        response = get_response(query)
    st.markdown(f"**Resposta:**\n\n{response}")
