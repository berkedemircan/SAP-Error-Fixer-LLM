out_dir = 'out_saplogs_v2'
eval_interval = 100
eval_iters = 200
log_interval = 10

always_save_checkpoint = True
wandb_log = False
wandb_project = 'saplogs_v2'
wandb_run_name = 'nano-m3pro-run'

dataset = 'saplogs'
gradient_accumulation_steps = 2
batch_size = 4
block_size = 384

n_layer = 4
n_head = 8
n_embd = 256
dropout = 0.12                  

learning_rate = 2e-4
max_iters = 4500
lr_decay_iters = 4500
min_lr = 3e-5

beta2 = 0.96                   
warmup_iters = 600              
decay_lr = True
weight_decay = 0.10             
grad_clip = 1.0
seed = 1337

backend = 'nccl'
device = 'mps'
compile = False

