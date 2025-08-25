import torch
import pickle
from model import GPT, GPTConfig

#Reading Prompt File
with open("data/saplogs/prompt_fewshot.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

#Tokenizer
with open("data/saplogs/vocab.pkl", "rb") as f:
    encoder = pickle.load(f)
    decoder = {v: k for k, v in encoder.items()}

def encode(s):
    return [encoder[c] for c in s if c in encoder]

def decode(tokens):
    return ''.join([decoder[t] for t in tokens])

#Encoding
encoded_prompt = encode(prompt)
idx = torch.tensor([encoded_prompt], dtype=torch.long).unsqueeze(0)  # shape: [1, T]

#Loading Model
ckpt = torch.load("out_saplogs_v2/ckpt.pt", map_location="cpu")
config = GPTConfig(**ckpt['model_args'])
model = GPT(config)
model.load_state_dict(ckpt['model'])
model.eval()


with torch.no_grad():
    out = model.generate(
        idx,
        max_new_tokens=300,
        temperature=0.7,
        top_k=40
    )


print("\n--- Prompt Completion Output ---\n")
print(decode(out[0].tolist()))
