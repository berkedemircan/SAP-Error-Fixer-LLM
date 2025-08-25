import pickle
import os


prompt_path = os.path.join("data", "saplogs", "prompt_fewshot.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    text = f.read()


chars = sorted(list(set(text)))
encoder = {ch: i for i, ch in enumerate(chars)}
decoder = {i: ch for ch, i in encoder.items()}


vocab_path = os.path.join("data", "saplogs", "vocab.pkl")
with open(vocab_path, "wb") as f:
    pickle.dump(encoder, f)

print("vocab.pkl created at:", vocab_path)

#Creating manual vocab.pkl file