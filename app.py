import streamlit as st
from llm_agent import get_response

st.set_page_config(page_title="Mentor LLM", layout="wide")
st.title("🤖 Mentor LLM")
st.markdown("_Seu orientador pessoal sobre Large Language Models, direto do futuro._")

query = st.text_input("❓ Pergunte qualquer coisa sobre LLMs")
if query:
    with st.spinner("Mentor pensando..."):
        response = get_response(query)
    st.markdown("### 📚 Resposta")
    st.write(response)
