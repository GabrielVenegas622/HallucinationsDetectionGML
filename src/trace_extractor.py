import torch
from transformers import pipeline
from huggingface_hub import login

# --- Including the token to access to llama ---
with open("llama_token.txt", "r") as file:
    t = file.read()
    login(token=t)

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    dtype=torch.bfloat16, 
    device_map="auto"
)

output = pipe("The key to life is")

print("--- Output Completo ---")
print(output)
