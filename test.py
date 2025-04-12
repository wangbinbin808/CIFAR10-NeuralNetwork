from model import ThreeLayerNN
from data_loader import load_CIFAR10, preprocess_data
from utils import compute_accuracy
import numpy as np

# 加载数据集
X_train_all, y_train_all, X_test, y_test = load_CIFAR10('./cifar-10-batches-py')
X_train_all, X_test = preprocess_data(X_train_all, X_test)

# 初始化模型时，指定两个隐藏层大小（必须与训练时一致）
model = ThreeLayerNN(input_size=3072, hidden_size1=128, hidden_size2=64, output_size=10)

# 加载训练好的参数
model_params = np.load('model_best.npz')
for k in model_params.files:
    model.params[k] = model_params[k]  # 加载 W1, b1, W2, b2, W3, b3

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算并输出准确率
acc = compute_accuracy(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")