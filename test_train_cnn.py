import mynn as nn
from draw_tools.plot import plot

import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from struct import unpack

np.random.seed(309)

train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))
with open('idx_cnn.pickle', 'wb') as f:
    pickle.dump(idx, f)

train_imgs = train_imgs[idx] / 255.0
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# Baseline + Dropout：simple CNN + 基础 SGD (无动量 mu=0)，不使用学习率调度
cnn_model = nn.models.Model_CNN(img_shape=(1, rows, cols), num_classes=train_labs.max() + 1, dropout_prob=0.2)

# 【改动点】将 mu 设为 0，关闭动量机制
optimizer = nn.optimizer.MomentGD(init_lr=0.03, model=cnn_model, mu=0.0)

loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=1,
    log_iters=100,
    save_dir='./best_cnn_models_dropout',
    checkpoint_dir='./saved_cnn_models_dropout',
    checkpoint_interval=1000
)

fig, axes = plt.subplots(1, 2)
fig.set_tight_layout(1)
plot(runner, axes)

os.makedirs('./figs', exist_ok=True)
fig.savefig('./figs/cnn_learning_curve_dropout.png', dpi=300)
plt.show()