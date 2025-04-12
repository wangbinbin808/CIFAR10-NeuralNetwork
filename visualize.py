import numpy as np
import matplotlib.pyplot as plt
import os

# 创建保存目录
os.makedirs("training_visualizations", exist_ok=True)
# 创建输出目录
os.makedirs('param_visualizations', exist_ok=True)

# 加载训练过程的历史记录
history = np.load("training_history.npz")
# 加载训练好的模型参数（假设保存在 best_model.npz）
params = np.load('model_best.npz')

# 提取训练历史
train_loss = history["train_loss"]
val_loss = history["val_loss"]
val_acc = history["val_acc"]
epochs = np.arange(1, len(train_loss) + 1)

# 图1：训练集和验证集的 Loss 曲线
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_visualizations/loss_curve.png")
plt.close()

# 图2：验证集 Accuracy 曲线
plt.figure(figsize=(8, 6))
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='^', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_visualizations/accuracy_curve.png")
plt.close()

# 遍历所有权重和偏置
for name in params.files:
    param = params[name]
    
    # 可视化方式取决于参数的维度
    if param.ndim == 2:
        # 权重矩阵（例如 W1, W2, W3）：画热力图
        plt.figure(figsize=(8, 6))
        plt.imshow(param, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f"{name} (shape={param.shape})")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.tight_layout()
        plt.savefig(f"param_visualizations/{name}_heatmap.png")
        plt.close()
    
    elif param.ndim == 1:
        # 偏置向量（例如 b1, b2, b3）：画折线图
        plt.figure(figsize=(6, 4))
        plt.plot(param, marker='o', linestyle='-')
        plt.title(f"{name} (shape={param.shape})")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(f"param_visualizations/{name}_plot.png")
        plt.close()
    
    else:
        # 如果参数维度 > 2，暂不支持自动可视化（可扩展）
        print(f"Skipping {name} (unsupported shape {param.shape})")