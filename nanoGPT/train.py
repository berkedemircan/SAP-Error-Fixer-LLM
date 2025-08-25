"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import json  # added for metrics

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import matplotlib
matplotlib.use("Agg")  # sunucuda/GUI olmadan kayıt için
import matplotlib.pyplot as plt

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
history = {
    "step": [],
    "train_loss": [],
    "val_loss": [],
    "ppl": [],
    "token_acc": [],  # 0-1 arası
    "f1_tok": []
}
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # recreate memmap every batch (avoid memory leak)
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size (and maybe decoder) from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
meta = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None) if isinstance(meta, dict) else None
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# GradScaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# compile
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# DDP wrap
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -------------------- eval helpers --------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    if decay_ratio < 0: decay_ratio = 0.0
    if decay_ratio > 1: decay_ratio = 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def _eval_val_metrics(eval_iters_local=None, max_samples_for_text=20):
    """
    Computes:
      - val/loss (mean), val/ppl
      - val/token_acc (argmax vs Y)
      - val/f1_token (ID set-overlap F1)
      - (if text decoding available) text/em, text/f1_word, text/distinct1, text/distinct2, (optional) text/bertscore_f1
    Text decoding strategies:
      1) If env USE_GPT2_TOKENIZER=1 or init_from startswith('gpt2'): use GPT2 tokenizer from transformers
      2) Else if meta has 'itos': join tokens heuristically
      3) Else: skip text metrics gracefully
    """
    iters = eval_iters if eval_iters_local is None else int(eval_iters_local)
    model.eval()
    losses = torch.zeros(iters)
    correct = 0
    total = 0
    preds_ids = []
    refs_ids = []

    for k in range(iters):
        X, Y = get_batch('val')
        with ctx:
            logits, loss = model(X, Y)  # logits: [B, T, V]
        losses[k] = loss.item()
        pred_ids = torch.argmax(logits, dim=-1)  # [B, T]
        correct += (pred_ids == Y).sum().item()
        total += Y.numel()

        # collect up to max_samples_for_text for sequence-level metrics
        remaining = max(0, max_samples_for_text - len(preds_ids))
        if remaining > 0:
            take = min(remaining, pred_ids.size(0))
            preds_ids.extend([pred_ids[i].detach().cpu().tolist() for i in range(take)])
            refs_ids.extend([Y[i].detach().cpu().tolist() for i in range(take)])

    model.train()
    mean_loss = float(losses.mean().item())
    ppl = math.exp(mean_loss)
    token_acc = correct / max(1, total)

    # token-id set F1
    def f1_on_ids(a, b):
        A, B = set(a), set(b)
        if not A and not B: return 1.0
        if not A or not B: return 0.0
        inter = len(A & B)
        prec = inter / len(A)
        rec = inter / len(B)
        return 0.0 if (prec + rec) == 0.0 else 2 * prec * rec / (prec + rec)

    f1_token = 0.0
    if preds_ids and refs_ids:
        f1s = [f1_on_ids(pi, ri) for pi, ri in zip(preds_ids, refs_ids)]
        f1_token = sum(f1s) / len(f1s)

    # attempt text decode
    def try_decode(list_of_ids):
        use_gpt2 = bool(os.environ.get('USE_GPT2_TOKENIZER')) or (isinstance(init_from, str) and init_from.startswith('gpt2'))
        if use_gpt2:
            try:
                from transformers import GPT2TokenizerFast
                tok = GPT2TokenizerFast.from_pretrained('gpt2')
                return [tok.decode(ids, skip_special_tokens=True) for ids in list_of_ids]
            except Exception:
                pass
        # meta->itos
        try:
            if isinstance(meta, dict) and 'itos' in meta and isinstance(meta['itos'], dict):
                itos = meta['itos']
                def join_tokens(ids):
                    return ''.join(itos.get(i, '') for i in ids)
                return [join_tokens(ids) for ids in list_of_ids]
        except Exception:
            pass
        return None

    text_pred = try_decode(preds_ids)
    text_ref  = try_decode(refs_ids)

    text_metrics = {}
    if text_pred is not None and text_ref is not None:
        def distinct_n(texts, n):
            uniq, total = set(), 0
            for t in texts:
                toks = t.split()
                total += max(0, len(toks) - n + 1)
                for i in range(len(toks) - n + 1):
                    uniq.add(tuple(toks[i:i+n]))
            return (len(uniq) / total) if total > 0 else 0.0

        d1 = distinct_n(text_pred, 1)
        d2 = distinct_n(text_pred, 2)

        def word_f1(p, r):
            P, R = set(p.lower().split()), set(r.lower().split())
            if not P and not R: return 1.0
            if not P or not R: return 0.0
            inter = len(P & R)
            prec = inter / len(P)
            rec  = inter / len(R)
            return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

        f1_word = sum(word_f1(p, r) for p, r in zip(text_pred, text_ref)) / max(1, len(text_pred))
        em = sum(1 for p, r in zip(text_pred, text_ref) if p.strip().lower() == r.strip().lower()) / max(1, len(text_pred))

        text_metrics.update({
            'text/distinct1': d1,
            'text/distinct2': d2,
            'text/f1_word': f1_word,
            'text/em': em,
        })

        # optional BERTScore (enable with env USE_BERTSCORE=1, BERTSCORE_LANG=tr/en/..)
        if os.environ.get('USE_BERTSCORE'):
            try:
                from bert_score import score as bert_score
                lang = os.environ.get('BERTSCORE_LANG', 'en')
                _, _, F1 = bert_score(text_pred, text_ref, lang=lang)
                text_metrics['text/bertscore_f1'] = float(F1.mean().item())
            except Exception as e:
                text_metrics['text/bertscore_error'] = str(e)

    return {
        'val/loss': mean_loss,
        'val/ppl': ppl,
        'val/token_acc': token_acc,
        'val/f1_token': f1_token,
        **text_metrics
    }
# ------------------ end eval helpers ------------------

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        


        

        # extra metrics (generate-free for ids; decode if possible for text)
        try:
            vm = _eval_val_metrics()
            print(f"[val] ppl {vm['val/ppl']:.2f} | token_acc {vm['val/token_acc']*100:.2f}% | f1_tok {vm['val/f1_token']:.3f}")
            if 'text/f1_word' in vm:
                add = f" | BERT {vm['text/bertscore_f1']:.3f}" if 'text/bertscore_f1' in vm else ""
                print(f"[text] EM {vm['text/em']*100:.2f}% | F1_word {vm['text/f1_word']:.3f} | d1 {vm['text/distinct1']:.3f} | d2 {vm['text/distinct2']:.3f}{add}")
            
            history["step"].append(int(iter_num))
            history["train_loss"].append(float(losses["train"]))
            history["val_loss"].append(float(vm.get("val/loss", losses["val"])))
            history["ppl"].append(float(vm["val/ppl"]) if vm.get("val/ppl") is not None else None)
            history["token_acc"].append(float(vm["val/token_acc"]) if vm.get("val/token_acc") is not None else None)
            history["f1_tok"].append(float(vm["val/f1_token"]) if vm.get("val/f1_token") is not None else None)
                
            
            # save json
            try:
                outj = {'iter': iter_num, 'lr': lr, 'mfu': running_mfu, **vm}
                with open(os.path.join(out_dir, f"val_metrics_{iter_num}.json"), "w", encoding="utf-8") as f:
                    json.dump(outj, f, ensure_ascii=False, indent=2)
                with open(os.path.join(out_dir, "val_metrics_last.json"), "w", encoding="utf-8") as f:
                    json.dump(outj, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print("[val metrics] save error:", e)
        except Exception as e:
            print("[val metrics] error:", e)
           

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
                "val/ppl": vm.get('val/ppl', None) if 'vm' in locals() else None,
                "val/token_acc": vm.get('val/token_acc', None) if 'vm' in locals() else None,
                "val/f1_token": vm.get('val/f1_token', None) if 'vm' in locals() else None,
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # train micro-steps
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing/logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
    
# --- SAVE JSON + PLOTS (ADD THIS BLOCK AT THE END) ---
if master_process:
    # JSON
    try:
        out_json = os.path.join(out_dir, "metrics_history.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[metrics] saved: {out_json}")
    except Exception as e:
        print("[metrics] save error:", e)

    # Tek metrik/tek eksen grafiği (senin stilinle)
    def _plot_single_axis(x, y, title, ylabel, filename, marker='s', linestyle='--'):
        # None değerleri filtrele
        xs, ys = [], []
        for xi, yi in zip(x, y):
            if yi is not None:
                xs.append(xi); ys.append(yi)
        if not ys:
            print(f"[plot] skip {title}: no data")
            return
        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, marker=marker, linestyle=linestyle, linewidth=1.8, label=title)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.legend(fontsize=10, loc="best")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        out_path = os.path.join("graphs", filename + ".png")
        #plt.savefig(os.path.join(out_dir, filename + ".pdf"), dpi=300, bbox_inches="tight") #for pdf formatted graph
        plt.savefig(os.path.join(out_dir, filename + ".png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved: {filename}.pdf/.png")

    steps = history["step"]

    # 1) Train Loss
    _plot_single_axis(steps, history["train_loss"],
                      title="Training Loss", ylabel="Loss",
                      filename="metric_train_loss", marker='o', linestyle='-')

    # 2) Validation Loss
    _plot_single_axis(steps, history["val_loss"],
                      title="Validation Loss", ylabel="Loss",
                      filename="metric_val_loss", marker='s', linestyle='--')

    # 3) Perplexity
    _plot_single_axis(steps, history["ppl"],
                      title="Perplexity", ylabel="Perplexity",
                      filename="metric_ppl", marker='s', linestyle='--')

    # 4) Token Accuracy (0–1 aralığı; yüzde istersen y-ekseni etiketi “Accuracy (%)” yapıp 100 ile çarparak y'yi dönüştürebilirsin)
    vals = [v*100 if v is not None else None for v in history["token_acc"]]
    _plot_single_axis(steps, history["token_acc"],
                      title="Token Accuracy", ylabel="Accuracy (%)",
                      filename="metric_token_acc", marker='s', linestyle='--')

    # 5) F1 Token Score
    _plot_single_axis(steps, history["f1_tok"],
                      title="F1 Token Score", ylabel="F1 (0–1)",
                      filename="metric_f1_tok", marker='s', linestyle='--')
# --- END SAVE JSON + PLOTS ---


#Made some configurations on default model file.