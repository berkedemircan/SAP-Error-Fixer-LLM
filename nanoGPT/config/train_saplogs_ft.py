out_dir = "out_saplogs_v2"

eval_interval = 100
eval_iters = 40
log_interval = 10
always_save_checkpoint = True

dataset = "saplogs"
gradient_accumulation_steps = 2
batch_size = 4
block_size = 128

learning_rate = 2e-4
max_iters = 3000
lr_decay_iters = 500
decay_lr = True

dropout = 0.1
wandb_log = False

init_from = "out_saplogs_v2"