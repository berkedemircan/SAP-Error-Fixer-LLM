import matplotlib.pyplot as plt

#Data
epochs = [1, 2, 3, 4, 5, 6]

val_loss   = [0.8279, 0.7521, 0.7141, 0.4469, 0.3206, 0.3295]


plt.figure(figsize=(7, 5))

#axis

plt.plot(epochs, val_loss, marker='s', linestyle='--', linewidth=1.8, label="Validation Loss")




plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Value 0–1 normalized validation loss", fontsize=12)
plt.title("Training Process Metrics", fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc="best")
plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

plt.tight_layout()
plt.savefig("egitim_metrikleri_single_axis.pdf", dpi=300, bbox_inches="tight")
plt.savefig("egitim_metrikleri_single_axis.png", dpi=300, bbox_inches="tight")

plt.show()
