from huggingface_hub import InferenceClient

# Substitua pelo seu token da Hugging Face, ou use variável de ambiente
HUGGINGFACE_TOKEN = "hf_..."  
client = InferenceClient(
    model="tiiuae/falcon-7b-instruct",
    token=HUGGINGFACE_TOKEN
)

def get_response(question: str) -> str:
    system_prompt = "Você é o Mentor LLM, um orientador especialista em modelos de linguagem."
    prompt = f"{system_prompt}\nUsuário: {question}\nMentor LLM:"
    
    response = client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    return response.strip()
