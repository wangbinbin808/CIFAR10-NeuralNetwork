import numpy as np
from model import ThreeLayerNN
from utils import compute_accuracy
from data_loader import load_CIFAR10, preprocess_data
import matplotlib.pyplot as plt
import os

# 超参数
hyperparams = {
    'hidden_size1': 128,
    'hidden_size2': 64,
    'activation': 'relu',
    'learning_rate': 0.1,
    'learning_rate_decay': 0.95,
    'reg': 1e-4,
    'num_epochs': 40,
    'batch_size': 200
}

# 加载并预处理数据
X_train_all, y_train_all, X_test, y_test = load_CIFAR10('./cifar-10-batches-py')
X_train_all, X_test = preprocess_data(X_train_all, X_test)

# 划分验证集
num_train = int(0.9 * X_train_all.shape[0])
X_train, y_train = X_train_all[:num_train], y_train_all[:num_train]
X_val, y_val = X_train_all[num_train:], y_train_all[num_train:]

# 初始化模型
model = ThreeLayerNN(input_size=3072,
                         hidden_size1=hyperparams['hidden_size1'],
                         hidden_size2=hyperparams['hidden_size2'],
                         output_size=10,
                         activation=hyperparams['activation'])


# 存储训练过程数据
train_loss_history = []
val_loss_history = []
val_acc_history = []
best_val_acc = 0
best_params = None

# 开始训练
for epoch in range(hyperparams['num_epochs']):
    # 打乱训练集顺序
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    for i in range(0, X_train.shape[0], hyperparams['batch_size']):
        X_batch = X_train[indices[i:i+hyperparams['batch_size']]]
        y_batch = y_train[indices[i:i+hyperparams['batch_size']]]

        # 前向 + 反向传播
        loss, probs = model.loss(X_batch, y_batch, hyperparams['reg'])
        grads = model.backward(y_batch, probs, hyperparams['reg'])
        model.update_params(grads, hyperparams['learning_rate'])

    # 每轮结束后评估一次性能
    train_loss, _ = model.loss(X_train, y_train, hyperparams['reg'])
    val_loss, val_probs = model.loss(X_val, y_val, hyperparams['reg'])
    val_pred = np.argmax(val_probs, axis=1)
    val_acc = compute_accuracy(y_val, val_pred)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    # 根据验证集准确率保存最优模型参数
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {k: v.copy() for k, v in model.params.items()}

    # 学习率衰减
    hyperparams['learning_rate'] *= hyperparams['learning_rate_decay']

# 保存最优模型
np.savez('model_best.npz', **best_params)
np.savez('training_history.npz', train_loss=train_loss_history, val_loss=val_loss_history, val_acc=val_acc_history)
