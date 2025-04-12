# CIFAR-10 三层神经网络分类器（使用 NumPy 实现）

本项目使用纯 NumPy 实现了一个标准的三层神经网络（包含两个隐藏层），用于在 CIFAR-10 数据集上进行图像分类任务。

## 项目结构

```
├── model.py                 # 三层神经网络模型
├── train.py                 # 模型训练脚本
├── test.py                  # 模型测试脚本
├── param_search.py          # 超参数网格搜索脚本
├── visualize.py             # 可视化损失与准确率曲线，模型参数的脚本
├── data_loader.py           # 数据加载与预处理
├── utils.py                 # 工具函数（准确率计算）
├── cifar-10-batches-py/     # CIFAR-10 数据集目录（需提前解压）
```

---

## 依赖环境

- Python 3.12
- NumPy
- Matplotlib


安装依赖（如果还未安装）：

```bash
pip install numpy matplotlib
```

---

## 数据准备

请先从 [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 下载 CIFAR-10 python 版本数据集并解压至项目根目录，解压后目录结构应如下：

```
cifar-10-batches-py/
    ├── data_batch_1
    ├── ...
    └── test_batch
```

---

## 训练模型

你可以使用如下命令训练模型：

```bash
python train.py --hidden_size1 128 --hidden_size2 64 --learning_rate 0.1 --reg 1e-4
```

可选参数说明：

- `--hidden_size1`：第一隐藏层神经元数
- `--hidden_size2`：第二隐藏层神经元数
- `--learning_rate`：学习率
- `--reg`：L2 正则化强度

训练过程会自动保存：

- 最优模型参数到 `model_best.npz`
- 训练日志（Loss/Accuracy）到 `training_history.npz`

---

## 测试模型

使用保存的最优模型测试测试集精度：

```bash
python test.py
```

---

## 可视化

运行以下命令以绘制训练过程的损失曲线、准确率曲线以及模型参数热力图：

```bash
python visualize.py
```

生成的图像将保存在：

- `training_visualizations/`：训练过程图（loss、accuracy）
- `param_visualizations/`：模型参数热力图、偏置折线图

---

## 超参数网格搜索

你可以运行以下命令进行网格搜索：

```bash
python param_search.py
```

该脚本会尝试多组隐藏层大小、学习率和正则化系数，并记录每组的验证准确率，结果将保存在：

- `hyperparameter_search_results.txt`

---