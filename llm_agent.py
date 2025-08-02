from huggingface_hub import InferenceClient

client = InferenceClient(
    model="tiiuae/falcon-7b-instruct",  # modelo que funciona com text-generation
    token=seu_token_aqui
)

def get_response(prompt):
    response = client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1
    )
    return response
