import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#config

MODEL_DIR = os.getenv("MODEL_DIR","/Users/berkedemircan/Documents/nanoGPT/out_saplogs_v2") 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_K = 200

#Building API

app = FastAPI()


#Connection to Front-End

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):  # React to API
    query: str

class GenerateResponse(BaseModel): # API to React
    response: str
    
#Global Variables
    
model = None
encode = None
decode = None
block_size = None
    
    
#Building NanoGPT model on Back-end

@app.on_event("startup")
def load_model():
    """Sunucu açılırken modeli ve tokenizer'ı 1 kez yükle."""
    global model, encode, decode, block_size

   
    meta_path = os.path.join("data", "saplogs", "meta.pkl") 
    meta = None
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

   
    from model import GPTConfig, GPT  
    ckpt_path = os.path.join(MODEL_DIR, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    model_args = checkpoint["model_args"]
    block_size_local = model_args["block_size"]
    gptconf = GPTConfig(**model_args)
    m = GPT(gptconf)
    state_dict = checkpoint["model"]
    
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    m.load_state_dict(state_dict, strict=True)
    m.to(DEVICE)
    m.eval()

   
    if meta and "stoi" in meta and "itos" in meta:
        
        stoi = meta["stoi"]
        itos = meta["itos"]
        def _encode(s: str):
            import numpy as np
            return [stoi.get(ch, 0) for ch in s]
        def _decode(ids):
            return "".join([itos[i] for i in ids])
        encode_fn, decode_fn = _encode, _decode
    else:
        
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def _encode(s: str):
            return enc.encode(s)
        def _decode(ids):
            return enc.decode(ids)
        encode_fn, decode_fn = _encode, _decode

   
    model = m
    encode = encode_fn
    decode = decode_fn
    block_size = block_size_local

    print(f"[startup] Model yüklendi: {MODEL_DIR} | device={DEVICE} | block_size={block_size}")

def generate_text(prompt: str) -> str:
    """Kısa bir generate yardımcı fonksiyonu."""
    assert model is not None, "Model yüklenemedi."
    import torch
    from torch.nn import functional as F

    x = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    
    x = x[:, -block_size:]

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits, _ = model(x)
            logits = logits[:, -1, :]
            if TEMPERATURE > 0:
                logits = logits / TEMPERATURE
                probs = F.softmax(logits, dim=-1)
                
                if TOP_K is not None:
                    v, ix = torch.topk(probs, min(TOP_K, probs.size(-1)))
                    probs2 = torch.zeros_like(probs).scatter_(1, ix, v)
                    probs = probs2 / probs2.sum(dim=-1, keepdim=True)
                ix = torch.multinomial(probs, num_samples=1)
            else:
                ix = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat((x, ix), dim=1)

    out_ids = x[0].tolist()
    
    return decode(out_ids)


@app.post("/generate")
def generate(req: GenerateRequest):
    user_text = req.query.strip()
    if not user_text:
        return {"response": "Boş giriş."}

    
    
    prompt = user_text

    try:
        text = generate_text(prompt)
        
        if text.startswith(prompt):
            text = text[len(prompt):].lstrip()
        return {"response": text}
    except Exception as e:
        print("generate error:", e)
        return {"response": "Model cevabı üretilirken hata oluştu."}
