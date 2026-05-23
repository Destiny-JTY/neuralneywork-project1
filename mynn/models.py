from .op import *
import numpy as np
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_prob=0.0):
        self.size_list = size_list
        self.act_func = act_func
        self.layers = []
        self.dropout_prob = dropout_prob
        self.training = True

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if self.dropout_prob > 0.0:
                        self.layers.append(Dropout(self.dropout_prob))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def set_train(self, mode=True):
        self.training = mode
        for layer in self.layers:
            if hasattr(layer, 'set_train'):
                layer.set_train(mode)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        # Backward compatibility:
        # older checkpoints store params from index 2, without dropout_prob.
        if len(param_list) > 2 and isinstance(param_list[2], (int, float)):
            self.dropout_prob = float(param_list[2])
            param_offset = 3
        else:
            self.dropout_prob = 0.0
            param_offset = 2
        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + param_offset]['W']
            layer.b = param_list[i + param_offset]['b']
            layer.params = {'W': layer.W, 'b': layer.b}
            layer.weight_decay = param_list[i + param_offset]['weight_decay']
            layer.weight_decay_lambda = param_list[i + param_offset]['lambda']
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplementedError
                elif self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                if self.dropout_prob > 0.0:
                    self.layers.append(Dropout(self.dropout_prob))
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func, self.dropout_prob]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, img_shape=(1, 28, 28), num_classes=10, lambda_list=None, dropout_prob=0.0):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.lambda_list = lambda_list
        self.input_shape = None
        self.feature_shape = None
        self.layers = []
        self.dropout_prob = dropout_prob
        self.training = True

        if img_shape is not None:
            self._build_layers()

    def _init(self, fan_in):
        return lambda size: np.random.normal(0, np.sqrt(2 / fan_in), size=size)

    def _build_layers(self):
        c, h, w = self.img_shape
        conv1 = conv2D(c, 8, 3, stride=2, padding=1, initialize_method=self._init(c * 3 * 3))
        conv2 = conv2D(8, 16, 3, stride=2, padding=1, initialize_method=self._init(8 * 3 * 3))
        h = (h + 2 * conv1.padding - conv1.kernel_size) // conv1.stride + 1
        w = (w + 2 * conv1.padding - conv1.kernel_size) // conv1.stride + 1
        h = (h + 2 * conv2.padding - conv2.kernel_size) // conv2.stride + 1
        w = (w + 2 * conv2.padding - conv2.kernel_size) // conv2.stride + 1
        fc = Linear(16 * h * w, self.num_classes, initialize_method=self._init(16 * h * w))
        self.layers = [conv1, ReLU()]
        if self.dropout_prob > 0.0:
            self.layers.append(Dropout(self.dropout_prob))
        self.layers.extend([conv2, ReLU()])
        if self.dropout_prob > 0.0:
            self.layers.append(Dropout(self.dropout_prob))
        self.layers.append(fc)
        if self.lambda_list is not None:
            for layer, lam in zip([conv1, conv2, fc], self.lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = lam

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        outputs = X.reshape(X.shape[0], *self.img_shape) if X.ndim == 2 else X
        for layer in self.layers[:-1]:
            outputs = layer(outputs)
        self.feature_shape = outputs.shape
        return self.layers[-1](outputs.reshape(outputs.shape[0], -1))

    def backward(self, loss_grad):
        grads = self.layers[-1].backward(loss_grad).reshape(self.feature_shape)
        for layer in reversed(self.layers[:-1]):
            grads = layer.backward(grads)
        return grads.reshape(self.input_shape)
    
    def set_train(self, mode=True):
        self.training = mode
        for layer in self.layers:
            if hasattr(layer, 'set_train'):
                layer.set_train(mode)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.img_shape = param_list['img_shape']
        self.num_classes = param_list['num_classes']
        self.lambda_list = None
        self.dropout_prob = param_list.get('dropout_prob', 0.0)
        self._build_layers()
        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, params in zip(optimizable_layers, param_list['params']):
            layer.params = {'W': params['W'], 'b': params['b']}
            layer.W, layer.b = layer.params['W'], layer.params['b']
            layer.weight_decay = params['weight_decay']
            layer.weight_decay_lambda = params['lambda']
        
    def save_model(self, save_path):
        params = []
        for layer in self.layers:
            if layer.optimizable:
                params.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump({'img_shape': self.img_shape, 'num_classes': self.num_classes, 'dropout_prob': self.dropout_prob, 'params': params}, f)
