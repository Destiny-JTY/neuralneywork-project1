from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.W, self.b = self.params['W'], self.params['b']
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = kernel_size
        self.stride, self.padding = stride, padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.W, self.b = self.params['W'], self.params['b']
        self.input = X
        Xp = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        n, _, h, w = Xp.shape
        k, s = self.kernel_size, self.stride
        out_h, out_w = (h - k) // s + 1, (w - k) // s + 1
        out = np.zeros((n, self.out_channels, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                window = Xp[:, :, i*s:i*s+k, j*s:j*s+k]
                out[:, :, i, j] = np.tensordot(window, self.W, axes=([1, 2, 3], [1, 2, 3]))
        return out + self.b

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        Xp = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        dXp = np.zeros_like(Xp)
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)
        k, s = self.kernel_size, self.stride
        for i in range(grads.shape[2]):
            for j in range(grads.shape[3]):
                hs, ws = i * s, j * s
                window = Xp[:, :, hs:hs+k, ws:ws+k]
                self.grads['W'] += np.tensordot(grads[:, :, i, j], window, axes=([0], [0]))
                dXp[:, :, hs:hs+k, ws:ws+k] += np.tensordot(grads[:, :, i, j], self.W, axes=([1], [0]))
        if self.padding == 0:
            return dXp
        return dXp[:, :, self.padding:-self.padding, self.padding:-self.padding]
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.grads = None
        self.predicts = None
        self.labels = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.labels = labels.astype(int)
        self.predicts = softmax(predicts) if self.has_softmax else predicts
        probs = np.clip(self.predicts[np.arange(labels.shape[0]), self.labels], 1e-12, 1.0)
        return -np.mean(np.log(probs))
    
    def backward(self):
        # first compute the grads from the loss to the input
        n = self.labels.shape[0]
        if self.has_softmax:
            self.grads = self.predicts.copy()
            self.grads[np.arange(n), self.labels] -= 1
        else:
            self.grads = np.zeros_like(self.predicts)
            self.grads[np.arange(n), self.labels] = -1 / np.clip(self.predicts[np.arange(n), self.labels], 1e-12, 1.0)
        self.grads /= n
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
