import matplotlib.pyplot as plt

#Data
epochs = [1, 2, 3, 4, 5, 6]
train_loss = [0.2520, 0.2278, 0.1573, 0.1100, 0.0522, 0.0548]
val_loss   = [0.8279, 0.7521, 0.7141, 0.4469, 0.3206, 0.3295]
ppl        = [2.27, 2.34, 1.72, 1.48, 1.37, 1.39]
token_acc  = [87.18, 86.76, 90.98, 93.27, 95.97, 95.32]  # percent
token_acc  = [v/100 for v in token_acc]  # converting token_acc to [0-1]
f1_tok     = [0.860, 0.905, 0.899, 0.907, 0.982, 0.982]  

plt.figure(figsize=(7, 5))

#axis
plt.plot(epochs, train_loss, marker='o', linestyle='-', linewidth=1.8, label="Train Loss")
plt.plot(epochs, val_loss, marker='s', linestyle='--', linewidth=1.8, label="Validation Loss")
plt.plot(epochs, ppl, marker='^', linestyle='-', linewidth=1.8, label="Perplexity (PPL)")
plt.plot(epochs, token_acc, marker='d', linestyle='-.', linewidth=1.8, label="Token Accuracy")
plt.plot(epochs, f1_tok, marker='x', linestyle=':', linewidth=1.8, label="F1 Token Score")

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Value (0–1 normalized or raw loss/ppl)", fontsize=12)
plt.title("Training Process Metrics", fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc="best")
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

plt.tight_layout()
plt.savefig("egitim_metrikleri_single_axis.pdf", dpi=300, bbox_inches="tight")
plt.savefig("egitim_metrikleri_single_axis.png", dpi=300, bbox_inches="tight")

plt.show()
