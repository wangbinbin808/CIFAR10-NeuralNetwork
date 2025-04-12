import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation='relu'):
        # 初始化三层神经网络的权重参数
        self.params = {
            'W1': 0.01 * np.random.randn(input_size, hidden_size1),
            'b1': np.zeros((1, hidden_size1)),
            'W2': 0.01 * np.random.randn(hidden_size1, hidden_size2),
            'b2': np.zeros((1, hidden_size2)),
            'W3': 0.01 * np.random.randn(hidden_size2, output_size),
            'b3': np.zeros((1, output_size))
        }
        self.activation = activation

    def _activation(self, x):
        # 支持 ReLU 和 Sigmoid 激活函数
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def _activation_grad(self, x):
        # 激活函数的导数，用于反向传播
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)

    def forward(self, X):
        # 前向传播：输入 -> 第一隐藏层 -> 第二隐藏层 -> 输出层
        z1 = X @ self.params['W1'] + self.params['b1']
        a1 = self._activation(z1)
        z2 = a1 @ self.params['W2'] + self.params['b2']
        a2 = self._activation(z2)
        z3 = a2 @ self.params['W3'] + self.params['b3']

        # 缓存中间变量，供反向传播使用
        self.cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3}
        return z3

    def loss(self, X, y, reg):
        # 计算 softmax + 交叉熵损失，同时加入 L2 正则化
        scores = self.forward(X)
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = X.shape[0]
        data_loss = -np.sum(log_probs[np.arange(N), y]) / N

        # 加入 L2 正则项，正则化所有权重参数
        reg_loss = 0.5 * reg * (
            np.sum(self.params['W1'] ** 2) +
            np.sum(self.params['W2'] ** 2) +
            np.sum(self.params['W3'] ** 2)
        )
        loss = data_loss + reg_loss
        return loss, probs

    def backward(self, y, probs, reg):
        # 手动实现反向传播过程，包含交叉熵损失和 L2 正则项
        grads = {}
        N = y.shape[0]

        dz3 = probs.copy()
        dz3[np.arange(N), y] -= 1
        dz3 /= N

        # 第三层梯度 + L2 正则
        grads['W3'] = self.cache['a2'].T @ dz3 + reg * self.params['W3']
        grads['b3'] = np.sum(dz3, axis=0, keepdims=True)

        # 反向传播到第二层
        da2 = dz3 @ self.params['W3'].T
        dz2 = da2 * self._activation_grad(self.cache['z2'])
        grads['W2'] = self.cache['a1'].T @ dz2 + reg * self.params['W2']
        grads['b2'] = np.sum(dz2, axis=0, keepdims=True)

        # 反向传播到第一层
        da1 = dz2 @ self.params['W2'].T
        dz1 = da1 * self._activation_grad(self.cache['z1'])
        grads['W1'] = self.cache['X'].T @ dz1 + reg * self.params['W1']
        grads['b1'] = np.sum(dz1, axis=0, keepdims=True)

        return grads

    def predict(self, X):
        # 预测类别标签
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def update_params(self, grads, learning_rate):
        # 使用 SGD 更新所有层的参数
        for key in self.params:
            self.params[key] -= learning_rate * grads[key]
