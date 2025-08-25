import matplotlib.pyplot as plt

#Data
epochs = [1, 2, 3, 4, 5, 6]

token_acc  = [87.18, 86.76, 90.98, 93.27, 95.97, 95.32]  


plt.figure(figsize=(7, 5))

#axis
plt.plot(epochs, token_acc, marker='d', linestyle='-.', linewidth=1.8, label="Token Accuracy")

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Value (0–1 normalized or raw loss/ppl)", fontsize=12)
plt.title("Training Process Metrics", fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc="best")
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

plt.tight_layout()
plt.savefig("egitim_metrikleri_single_axis.pdf", dpi=300, bbox_inches="tight")
plt.savefig("egitim_metrikleri_single_axis.png", dpi=300, bbox_inches="tight")

plt.show()
