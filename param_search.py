import itertools
import os
import numpy as np

# 网格搜索超参数组合
hidden_sizes1 = [256, 128]  # 第一个隐藏层的大小
hidden_sizes2 = [64, 32]   # 第二个隐藏层的大小
learning_rates = [0.1, 0.01]  # 学习率
regs = [1e-3, 1e-4]      # 正则化强度

# 打开一个txt文件用于保存结果
with open('hyperparameter_search_results.txt', 'w') as f:
    # 写入表头
    f.write("hidden_size1\thidden_size2\tlearning_rate\treg\tval_accuracy\n")
    
    # 结果记录列表
    search_results = []

    for hs1, hs2, lr, reg in itertools.product(hidden_sizes1, hidden_sizes2, learning_rates, regs):
        # 训练模型时传入当前超参数
        cmd = f"python train.py --hidden_size1 {hs1} --hidden_size2 {hs2} --learning_rate {lr} --reg {reg}"
        print(f"Running: {cmd}")

        # 执行训练命令
        os.system(cmd)

        # 加载保存的训练过程数据
        history = np.load('training_history.npz')
        val_acc = history['val_acc'][-1]  # 记录最后一个 epoch 的验证集准确率

        # 将当前超参数组合和验证准确率写入txt文件
        f.write(f"{hs1}\t{hs2}\t{lr}\t{reg}\t{val_acc:.4f}\n")

        # 输出每次训练的结果
        print(f"Val Accuracy for (hidden_size1={hs1}, hidden_size2={hs2}, learning_rate={lr}, reg={reg}): {val_acc:.4f}")

        # 保存当前超参数与其性能结果
        search_results.append({
            'hidden_size1': hs1,
            'hidden_size2': hs2,
            'learning_rate': lr,
            'reg': reg,
            'val_acc': val_acc
        })

    # 搜索完成后提示
    print("Hyperparameter search complete. Results saved to 'hyperparameter_search_results.txt'.")