import gzip
import os
import pickle
import re
from struct import unpack

os.environ.setdefault('MPLCONFIGDIR', './.matplotlib')

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


MLP_MODEL = './best_models/best_model.pickle'
CNN_MODEL = './best_cnn_models/best_model.pickle'
MLP_CKPT_DIR = './saved_models'
CNN_CKPT_DIR = './saved_cnn_models'
FIG_DIR = './figs/comparison'
TEST_IMAGES = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
TEST_LABELS = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'
TRAIN_IMAGES = './dataset/MNIST/train-images-idx3-ubyte.gz'
TRAIN_LABELS = './dataset/MNIST/train-labels-idx1-ubyte.gz'
VALID_INDEX = './idx.pickle'


def load_mnist(images_path, labels_path):
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols) / 255.0
    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


def load_validation_set():
    images, labels = load_mnist(TRAIN_IMAGES, TRAIN_LABELS)
    if os.path.exists(VALID_INDEX):
        with open(VALID_INDEX, 'rb') as f:
            idx = pickle.load(f)
    else:
        np.random.seed(309)
        idx = np.random.permutation(np.arange(images.shape[0]))
    return images[idx][:10000], labels[idx][:10000]


def predict(model, X, batch_size=512):
    return np.vstack([model(X[i:i + batch_size]) for i in range(0, X.shape[0], batch_size)])


def evaluate(model, X, y):
    logits = predict(model, X)
    loss = nn.op.MultiCrossEntropyLoss()(logits, y)
    preds = np.argmax(logits, axis=1)
    return loss, np.mean(preds == y), preds


def load_model(kind, path):
    model = nn.models.Model_MLP() if kind == 'MLP' else nn.models.Model_CNN()
    model.load_model(path)
    return model


def count_params(model):
    total = 0
    for layer in model.layers:
        if layer.optimizable:
            total += sum(param.size for param in layer.params.values())
    return total


def confusion_matrix(y_true, y_pred, num_classes=10):
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        mat[true, pred] += 1
    return mat


def per_class_accuracy(mat):
    return np.diag(mat) / np.maximum(mat.sum(axis=1), 1)


def checkpoint_files(path):
    if not os.path.isdir(path):
        return []

    def step(name):
        match = re.search(r'checkpoint_step_(\d+)\.pickle$', name)
        return int(match.group(1)) if match else -1

    names = [name for name in os.listdir(path) if step(name) >= 0]
    return [(step(name), os.path.join(path, name)) for name in sorted(names, key=step)]


def evaluate_checkpoints(kind, ckpt_dir, X, y):
    rows = []
    for step, path in checkpoint_files(ckpt_dir):
        loss, acc, _ = evaluate(load_model(kind, path), X, y)
        rows.append((step, loss, acc))
        print(f'{kind} checkpoint {step}: loss={loss:.4f}, acc={acc:.4f}')
    return rows


def save_summary_csv(rows):
    path = os.path.join(FIG_DIR, 'summary.csv')
    with open(path, 'w') as f:
        f.write('model,test_loss,test_accuracy,parameters\n')
        for row in rows:
            f.write(f"{row['name']},{row['loss']:.6f},{row['acc']:.6f},{row['params']}\n")


def plot_summary(rows):
    names = [row['name'] for row in rows]
    accs = [row['acc'] for row in rows]
    losses = [row['loss'] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, accs, color=['#4C78A8', '#F58518'])
    axes[0].set_ylabel('Test accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].bar_label(axes[0].containers[0], fmt='%.3f')

    axes[1].bar(names, losses, color=['#4C78A8', '#F58518'])
    axes[1].set_ylabel('Test loss')
    axes[1].bar_label(axes[1].containers[0], fmt='%.3f')

    fig.suptitle('MLP vs CNN on MNIST Test Set')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'test_summary.png'), dpi=300)
    plt.close(fig)


def plot_checkpoint_curves(mlp_rows, cnn_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, rows, color in [('MLP', mlp_rows, '#4C78A8'), ('CNN', cnn_rows, '#F58518')]:
        if rows:
            steps, losses, accs = zip(*rows)
            axes[0].plot(steps, losses, marker='o', label=name, color=color)
            axes[1].plot(steps, accs, marker='o', label=name, color=color)

    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Validation loss')
    axes[0].legend()
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Validation accuracy')
    axes[1].legend()
    fig.suptitle('Checkpoint Performance on Shared Validation Set')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'checkpoint_validation_curves.png'), dpi=300)
    plt.close(fig)


def plot_confusions(mats):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, mat) in zip(axes, mats.items()):
        image = ax.imshow(mat, cmap='Blues')
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle('Confusion Matrices on Test Set')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'confusion_matrices.png'), dpi=300)
    plt.close(fig)


def plot_per_class(mats):
    x = np.arange(10)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, per_class_accuracy(mats['MLP']), width, label='MLP', color='#4C78A8')
    ax.bar(x + width / 2, per_class_accuracy(mats['CNN']), width, label='CNN', color='#F58518')
    ax.set_xlabel('Digit class')
    ax.set_ylabel('Test accuracy')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'per_class_accuracy.png'), dpi=300)
    plt.close(fig)


def plot_examples(X, y, preds, name):
    wrong = np.where(preds != y)[0][:16]
    correct = np.where(preds == y)[0][:16]
    selected = wrong if len(wrong) else correct

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, idx in zip(axes.ravel(), selected):
        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'T:{y[idx]} P:{preds[idx]}')
        ax.axis('off')
    fig.suptitle(f'{name} Prediction Examples')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'{name.lower()}_examples.png'), dpi=300)
    plt.close(fig)


def plot_conv_kernels(model):
    conv_layers = [layer for layer in model.layers if isinstance(layer, nn.op.conv2D)]
    for layer_idx, layer in enumerate(conv_layers, start=1):
        weights = layer.params['W']
        out_channels, in_channels, _, _ = weights.shape
        cols = min(out_channels, 8)
        rows = int(np.ceil(out_channels / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
        axes = np.array(axes).reshape(-1)

        for i, ax in enumerate(axes):
            ax.axis('off')
            if i < out_channels:
                kernel = weights[i].mean(axis=0) if in_channels > 1 else weights[i, 0]
                ax.imshow(kernel, cmap='gray')
                ax.set_title(f'K{i}', fontsize=8)

        fig.suptitle(f'CNN Conv Layer {layer_idx} Kernels')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'cnn_conv{layer_idx}_kernels.png'), dpi=300)
        plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    test_X, test_y = load_mnist(TEST_IMAGES, TEST_LABELS)
    valid_X, valid_y = load_validation_set()

    models = {'MLP': load_model('MLP', MLP_MODEL), 'CNN': load_model('CNN', CNN_MODEL)}
    rows, mats, preds_by_model = [], {}, {}

    for name, model in models.items():
        loss, acc, preds = evaluate(model, test_X, test_y)
        rows.append({'name': name, 'loss': loss, 'acc': acc, 'params': count_params(model)})
        mats[name] = confusion_matrix(test_y, preds)
        preds_by_model[name] = preds
        print(f"{name}: test_loss={loss:.4f}, test_acc={acc:.4f}, params={count_params(model)}")

    mlp_ckpts = evaluate_checkpoints('MLP', MLP_CKPT_DIR, valid_X, valid_y)
    cnn_ckpts = evaluate_checkpoints('CNN', CNN_CKPT_DIR, valid_X, valid_y)

    save_summary_csv(rows)
    plot_summary(rows)
    plot_checkpoint_curves(mlp_ckpts, cnn_ckpts)
    plot_confusions(mats)
    plot_per_class(mats)
    plot_conv_kernels(models['CNN'])
    for name, preds in preds_by_model.items():
        plot_examples(test_X, test_y, preds, name)

    print(f'Comparison figures and summary saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
