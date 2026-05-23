import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os
import pickle

# 固定随机种子
np.random.seed(309)

train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 划分验证集
idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 归一化
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# Baseline + Dropout：MLP + 基础 SGD (无动量 mu=0)，不使用学习率调度
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', dropout_prob=0.2)

# 【改动点】将 mu 设为 0，让其退化为不带动量的纯 SGD 优化器
optimizer = nn.optimizer.MomentGD(init_lr=0.03, model=linear_model, mu=0.0) 

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=1,
    log_iters=100,
    save_dir='./best_models_dropout',
    checkpoint_dir='./saved_models_dropout',
    checkpoint_interval=1000
)

fig, axes = plt.subplots(1, 2)
fig.set_tight_layout(1)
plot(runner, axes)

os.makedirs('./figs', exist_ok=True)
fig.savefig('./figs/learning_curve_dropout.png', dpi=300)
plt.show()