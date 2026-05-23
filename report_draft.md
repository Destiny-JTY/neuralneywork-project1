# Project 1 Report Draft: MNIST Classification with MLP and CNN

GitHub link: TODO

## 1. MLP Baseline

In Part A, I implemented the basic components required for a multi-layer perceptron from scratch using NumPy. The implemented components include the forward and backward propagation of a linear layer, softmax, and multi-class cross-entropy loss. No deep learning framework such as PyTorch or TensorFlow was used.

The MLP baseline uses one hidden layer:

```text
Input 784 -> Linear 600 -> ReLU -> Linear 10
```

The input images are flattened MNIST images of shape `28 * 28 = 784`. The model is trained on 50,000 training samples, while 10,000 samples from the original training set are used as the validation set. The final test accuracy is evaluated on the official MNIST test set.

Baseline training setting:

```text
optimizer: SGD
learning rate: 0.01
batch size: 32
epochs: 2
regularization: none
momentum: none
learning rate scheduling: none
```

The MLP baseline reaches:

```text
validation accuracy: 0.8944
test accuracy:       0.8941
test loss:           2.2677
number of params:    477010
```

The learning curve is saved at:

```text
figs/learning_curve.png
```

## 2. CNN Model and MLP-vs-CNN Comparison

In Part B, I implemented a simple CNN model using the self-written `conv2D` operator. The CNN keeps the same dataset split and training protocol as the MLP baseline so that the comparison is reasonably fair.

The CNN architecture is:

```text
Input 1 x 28 x 28
-> Conv2D 1 -> 8, kernel 3, stride 2, padding 1
-> ReLU
-> Conv2D 8 -> 16, kernel 3, stride 2, padding 1
-> ReLU
-> Flatten
-> Linear 16 * 7 * 7 -> 10
```

Baseline CNN training setting:

```text
optimizer: SGD
learning rate: 0.01
batch size: 32
epochs: 2
regularization: none
momentum: none
learning rate scheduling: none
```

The CNN baseline reaches:

```text
validation accuracy: 0.9167
test accuracy:       0.9219
test loss:           0.2579
number of params:    9098
```

Compared with the MLP baseline, the CNN achieves higher validation and test accuracy while using far fewer parameters. This suggests that the convolutional inductive bias is useful for MNIST. Unlike the MLP, which treats every pixel position as an unrelated input dimension, the CNN uses local receptive fields and shared filters. This makes it better suited for image data, where nearby pixels form meaningful local patterns such as strokes, corners, and edges.

The CNN learning curve is saved at:

```text
figs/cnn_learning_curve.png
```

The MLP-vs-CNN comparison figures are saved under:

```text
figs/comparison/
```

Important figures include:

```text
figs/comparison/test_summary.png
figs/comparison/checkpoint_validation_curves.png
figs/comparison/confusion_matrices.png
figs/comparison/per_class_accuracy.png
figs/comparison/cnn_conv1_kernels.png
figs/comparison/cnn_conv2_kernels.png
figs/comparison/mlp_examples.png
figs/comparison/cnn_examples.png
```

## 3. Two Additional Directions

### 3.1 Direction 1: Optimization

For the optimization direction, I tried momentum SGD together with learning rate scheduling. The modified optimizer setting is:

```text
optimizer: Momentum SGD
initial learning rate: 0.03
momentum coefficient: 0.9
scheduler: MultiStepLR
milestones: [800, 2400, 4000]
gamma: 0.5
```

The optimized MLP model reaches:

```text
validation accuracy: 0.9280
test accuracy:       0.9278
test loss:           0.7615
```

The optimized CNN model reaches:

```text
validation accuracy: 0.9685
test accuracy:       0.9714
test loss:           0.0871
```

Both models improve after the optimization modification. The CNN benefits more clearly, improving from 0.9219 test accuracy to 0.9714 test accuracy. Momentum helps the optimizer move more consistently along useful gradient directions, while the learning rate schedule reduces the step size later in training to stabilize convergence.

Note: for a stricter experimental conclusion, momentum and learning rate scheduling can be studied separately. The current experiment treats them as one combined optimization setting.

Optimized learning curves are saved at:

```text
figs/learning_curve_optimized.png
figs/cnn_learning_curve_optimized.png
```

### 3.2 Direction 2: Regularization

Chosen direction: L2 regularization.

The codebase already supports L2-style weight decay through the `weight_decay` and `weight_decay_lambda` fields in optimizable layers. For example, `Model_MLP` and `Model_CNN` can be constructed with non-empty `lambda_list` values so that linear and convolutional weights are regularized during optimization.

Current status: the regularized model result has not yet been generated in the saved model folders. To complete the final report, train a regularized version under the same setting as the corresponding baseline, changing only the L2 regularization strength. The final table should then include the validation/test accuracy of the regularized model.

Suggested regularization setting:

```text
optimizer: SGD
learning rate: 0.01
batch size: 32
epochs: 2
regularization: L2 weight decay
lambda: 1e-4
momentum: none
learning rate scheduling: none
```

Expected comparison:

```text
MLP baseline vs MLP + L2
or
CNN baseline vs CNN + L2
```

This direction should answer whether L2 regularization improves validation/test accuracy or mainly reduces overfitting.

## 4. Main Results Table

| Model | Setting | Params | Val Loss | Val Acc | Test Loss | Test Acc |
|---|---|---:|---:|---:|---:|---:|
| MLP | baseline SGD | 477010 | 2.3466 | 0.8944 | 2.2677 | 0.8941 |
| CNN | baseline SGD | 9098 | 0.2827 | 0.9167 | 0.2579 | 0.9219 |
| MLP | momentum + LR schedule | 477010 | 0.7687 | 0.9280 | 0.7615 | 0.9278 |
| CNN | momentum + LR schedule | 9098 | 0.1026 | 0.9685 | 0.0871 | 0.9714 |
| MLP/CNN | L2 regularization | TODO | TODO | TODO | TODO | TODO |

The most important result is that the CNN baseline already outperforms the MLP baseline despite having far fewer parameters. This supports the conclusion that convolution is more suitable for image classification than a fully connected model on flattened pixels.

## 5. Detailed Visualization

The report can include the following visualizations:

1. Learning curves:

```text
figs/learning_curve.png
figs/cnn_learning_curve.png
figs/learning_curve_optimized.png
figs/cnn_learning_curve_optimized.png
```

2. MLP-vs-CNN comparison:

```text
figs/comparison/test_summary.png
figs/comparison/checkpoint_validation_curves.png
```

3. Confusion matrices:

```text
figs/comparison/confusion_matrices.png
```

4. Per-class accuracy:

```text
figs/comparison/per_class_accuracy.png
```

5. Convolution kernels:

```text
figs/comparison/cnn_conv1_kernels.png
figs/comparison/cnn_conv2_kernels.png
```

6. Prediction examples:

```text
figs/comparison/mlp_examples.png
figs/comparison/cnn_examples.png
```

The confusion matrix and prediction examples are especially useful for discussing which digits remain difficult. Similar-looking digit pairs, such as 4/9, 3/5, or 7/9, are typically harder because their handwritten shapes can overlap depending on writing style.

## 6. Discussion

### Why is CNN more suitable than MLP for image classification?

CNNs are more suitable because they preserve the spatial structure of images. A convolution filter only looks at a local region, so it can learn local visual patterns such as strokes, corners, and edges. The same filter is shared across different spatial positions, which means the CNN can detect similar features no matter where they appear in the image. In contrast, the MLP receives a flattened vector and does not explicitly model local neighborhoods or translation-like patterns.

### Does the CNN improve validation or test accuracy?

Yes. The CNN baseline improves both validation and test accuracy:

```text
MLP baseline test accuracy: 0.8941
CNN baseline test accuracy: 0.9219
```

The CNN also uses far fewer parameters:

```text
MLP params: 477010
CNN params: 9098
```

This is an important result: the CNN is not only more accurate, but also much more parameter-efficient.

### Which two additional directions were chosen?

The two chosen directions are:

```text
1. Optimization: momentum SGD + learning rate scheduling
2. Regularization: L2 weight decay
```

The optimization experiment has already been run and shows clear improvements for both MLP and CNN. The L2 regularization experiment still needs to be run to fill in the final table.

### Which modification or analysis is the most informative?

The MLP-vs-CNN comparison is the most informative because it demonstrates the importance of model structure. Even without optimization tricks, the CNN outperforms the MLP and uses far fewer parameters. This directly supports the course concept that CNNs are well matched to image data.

The optimization experiment is also informative because it shows that training strategy can significantly improve final performance. However, since the current optimization experiment changes both momentum and learning rate scheduling, it is less isolated than the baseline MLP-vs-CNN comparison.

### What kinds of samples are still hard for the model?

The difficult samples are usually ambiguous handwriting examples. Digits with similar shapes are more likely to be confused, especially when strokes are missing, connected, or unusually slanted. The prediction example figures and confusion matrices should be used to identify the most common mistakes in the final report.

