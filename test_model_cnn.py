import mynn as nn
import gzip
import numpy as np
from struct import unpack

model = nn.models.Model_CNN()
model.load_model('./best_cnn_models/best_model.pickle')

test_images_path = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / 255.0
logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))
