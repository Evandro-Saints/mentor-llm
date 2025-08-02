import streamlit as st
from llm_agent import get_response

st.set_page_config(page_title="🤖 Mentor LLM", page_icon="🧠")

st.title("🤖 Mentor LLM")
st.caption("Seu orientador pessoal sobre Large Language Models, direto do futuro.")

st.markdown("❓ **Pergunte qualquer coisa sobre LLMs**")

query = st.text_input("Digite sua pergunta aqui:", placeholder="Ex: O que é fine-tuning?")

if st.button("Enviar") and query:
    with st.spinner("Pensando como um LLM..."):
        try:
            response = get_response(query)
            st.markdown("### Resposta:")
            st.write(response)
        except Exception as e:
            st.error(f"Ocorreu um erro: {str(e)}")
