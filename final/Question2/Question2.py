import numpy as np
from collections import namedtuple
import copy
import pickle
import time

Batch = namedtuple('Batch', ['inputs', 'targets'])


class Initializer:
    def __call__(self, shape):
        return self.init(shape).astype(np.float32)

    def init(self, shape):
        raise NotImplementedError


class XavierUniform(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def init(self, shape):
        if len(shape) == 2:
            f_in = shape[0]
            f_out = shape[1]
        else:
            f_in = np.prod(shape[1:])
            f_out = shape[0]

        f_out = shape[1] if len(shape) == 2 else shape[0]
        a = self.gain * np.sqrt(6.0 / (f_in + f_out))
        return np.random.uniform(low=-a, high=a, size=shape)


class Zeros(Initializer):
    def init(self, shape):
        return np.zeros(shape=shape, dtype=np.float32)


class Ones(Initializer):
    def init(self, shape):
        return np.ones(shape=shape, dtype=np.float32)


class Layer:
    def __init__(self):
        self.parameters = {}
        for p in self.para_names:
            self.parameters[p] = None
        self.u_parameters = {}
        for p in self.u_para_names:
            self.u_parameters[p] = None

        self.gradients = {}
        self.shapes = {}
        self.is_training = True
        self.is_init = False

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, gradients):
        raise NotImplementedError

    def set_phase(self, phase):
        if phase == 'Train':
            self.is_training = True
        else:
            self.is_training = False

    def get_padding_1d(self, w, k):
        if self.padding_mode == 'SAME':
            pads = (w - 1) + k - w
            half = pads // 2
            padding = (half, half) if pads % 2 == 0 else (half, half + 1)
        else:
            padding = (0, 0)
        return padding

    def get_padding_2d(self, in_shape, k_shape):
        h_pad = self.get_padding_1d(in_shape[0], k_shape[0])
        w_pad = self.get_padding_1d(in_shape[1], k_shape[1])
        return (0, 0), h_pad, w_pad, (0, 0)

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return 'Layer: %s shape: %s' % self.name, self.shapes

    @property
    def para_names(self):
        return []

    @property
    def u_para_names(self):
        return []


class Dense(Layer):
    def __init__(self,
                 dim_out,
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.initializers = {'w': w_init, 'b': b_init}
        self.shapes = {'w': [None, dim_out], 'b': [dim_out]}

        self.inputs = None

    def forward(self, inputs):
        if not self.is_init:
            self.shapes['w'][0] = inputs.shape[1]
            self.init_paras()
        self.inputs = inputs
        return inputs @ self.parameters['w'] + self.parameters['b']

    def backward(self, grad):
        self.gradients['w'] = self.inputs.T @ grad
        self.gradients['b'] = np.sum(grad, axis=0)
        return grad @ self.parameters['w'].T

    def init_paras(self):
        for p in self.para_names:
            self.parameters[p] = self.initializers[p](self.shapes[p])
        self.is_init = True

    @property
    def para_names(self):
        return 'w', 'b'


class Conv2D(Layer):
    def __init__(self,
                 kernel,
                 stride=(1, 1),
                 padding='SAME',
                 w_init=XavierUniform(),
                 b_init=Zeros()):
        super().__init__()

        self.kernel_shape = kernel
        self.stride = stride
        self.initializers = {'w': w_init, 'b': b_init}
        self.shapes = {'w': self.kernel_shape, 'b': self.kernel_shape[-1]}

        self.padding_mode = padding
        self.padding = None

    def im2col(self, img, k_h, k_w, s_h, s_w):
        batch_sz, h, w, in_c = img.shape
        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1
        col = np.empty((batch_sz * out_h * out_w, k_h * k_w * in_c))
        batch_span = out_w * out_h
        for r in range(out_h):
            r_start = r * s_h
            matrix_r = r * out_w
            for c in range(out_w):
                c_start = c * s_w
                patch = img[:, r_start: r_start + k_h, c_start: c_start + k_w, :]
                patch = patch.reshape(batch_sz, -1)
                col[matrix_r + c::batch_span, :] = patch
        return col

    def forward(self, inputs):
        if not self.is_init:
            self.init_paras()

        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        X = self.inputs_preprocess(inputs)
        col = self.im2col(X, k_h, k_w, s_h, s_w)
        W = self.parameters['w'].reshape(-1, out_c)
        Z = col @ W
        batch_sz, in_h, in_w, _ = X.shape
        Z = Z.reshape(batch_sz, Z.shape[0] // batch_sz, out_c)
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        Z = Z.reshape(batch_sz, out_h, out_w, out_c)
        Z += self.parameters['b']
        self.X_shape, self.col, self.W = X.shape, col, W
        return Z

    def backward(self, grad):
        k_h, k_w, in_c, out_c = self.kernel_shape
        s_h, s_w = self.stride
        batch_sz, in_h, in_w, in_c = self.X_shape
        pad_h, pad_w = self.padding[1:3]
        flat_grad = grad.reshape((-1, out_c))
        d_W = self.col.T @ flat_grad
        self.gradients['w'] = d_W.reshape(self.kernel_shape)
        self.gradients['b'] = np.sum(flat_grad, axis=0)

        d_X = grad @ self.W.T
        d_in = np.zeros(shape=self.X_shape)
        for i, r in enumerate(range(0, in_h - k_h + 1, s_h)):
            for j, c in enumerate(range(0, in_w - k_w + 1, s_w)):
                patch = d_X[:, i, j, :]
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r:r + k_h, c:c + k_w, :] += patch

        d_in = d_in[:, pad_h[0]:in_h - pad_h[1], pad_w[0]:in_w - pad_w[1], :]
        return self.grads_postprocess(d_in)

    def inputs_preprocess(self, inputs):
        _, in_h, in_w, _ = inputs.shape
        k_h, k_w, _, _ = self.kernel_shape
        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w))
        return np.pad(inputs, pad_width=self.padding, mode='constant')

    def grads_postprocess(self, grads):
        return grads

    def init_paras(self):
        for p in self.para_names:
            self.parameters[p] = self.initializers[p](self.shapes[p])
        self.is_init = True

    @property
    def para_names(self):
        return 'w', 'b'


class MaxPool2D(Layer):
    def __init__(self, pool_size, stride, padding='VALID'):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

        self.padding_mode = padding
        self.padding = None

    def forward(self, inputs):
        s_h, s_w = self.stride
        k_h, k_w = self.pool_size
        batch_sz, in_h, in_w, in_c = inputs.shape

        if self.padding is None:
            self.padding = self.get_padding_2d(
                (in_h, in_w), (k_h, k_w))
        X = np.pad(inputs, pad_width=self.padding, mode='constant')
        padded_h, padded_w = X.shape[1:3]

        out_h = (padded_h - k_h) // s_h + 1
        out_w = (padded_w - k_w) // s_w + 1

        max_pool = np.empty(shape=(batch_sz, out_h, out_w, in_c))
        argmax = np.empty(shape=(batch_sz, out_h, out_w, in_c), dtype=int)
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                pool = X[:, r_start: r_start + k_h, c_start: c_start + k_w, :]
                pool = pool.reshape((batch_sz, -1, in_c))

                _argmax = np.argmax(pool, axis=1)[:, np.newaxis, :]
                argmax[:, r, c, :] = _argmax.squeeze()

                _max_pool = np.take_along_axis(pool, _argmax, axis=1).squeeze()
                max_pool[:, r, c, :] = _max_pool

        self.X_shape = X.shape
        self.out_shape = (out_h, out_w)
        self.argmax = argmax
        return max_pool

    def backward(self, grad):
        batch_sz, in_h, in_w, in_c = self.X_shape
        out_h, out_w = self.out_shape
        s_h, s_w = self.stride
        k_h, k_w = self.pool_size
        k_sz = k_h * k_w
        pad_h, pad_w = self.padding[1:3]

        d_in = np.zeros(shape=(batch_sz, in_h, in_w, in_c))
        for r in range(out_h):
            r_start = r * s_h
            for c in range(out_w):
                c_start = c * s_w
                _argmax = self.argmax[:, r, c, :]
                mask = np.eye(k_sz)[_argmax].transpose((0, 2, 1))
                _grad = grad[:, r, c, :][:, np.newaxis, :]
                patch = np.repeat(_grad, k_sz, axis=1) * mask
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r_start:r_start + k_h, c_start:c_start + k_w, :] += patch

        d_in = d_in[:, pad_h[0]:in_h - pad_h[1], pad_w[0]:in_w - pad_w[1], :]
        return d_in


class Reshape(Layer):
    def __init__(self, *output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], *self.output_shape)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class Activation(Layer):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        return self.func(x) * (1.0 - self.func(x))


class Relu(Activation):
    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative(self, x):
        return x > 0.0


class Loss:
    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, *args, **kwargs):
        raise NotImplementedError


class L1(Loss):
    def loss(self, predicted, actual):
        return np.mean(np.abs(predicted - actual))

    def grad(self, predicted, actual):
        b, h, w, c = predicted.shape
        return np.sign(predicted - actual) / (b * h * w * c)


class L2(Loss):
    def loss(self, predicted, actual):
        return 0.5 * np.sum((predicted - actual) ** 2) / predicted.shape[0]

    def grad(self, predicted, actual):
        return (predicted - actual) / predicted.shape[0]


class Optimizer:
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, grads, params):
        grad_values = grads.values
        step_values = self.compute_step(grad_values)
        grads.values = step_values

        if self.weight_decay:
            grads -= self.lr * self.weight_decay * params
        
        params += grads

    def compute_step(self, grad):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.b1 = beta1
        self.b2 = beta2
        self.eps = epsilon

        self.t = 0
        self.m = 0
        self.v = 0

    def compute_step(self, grad):
        self.t += 1

        self.m += (1.0 - self.b1) * (grad - self.m)
        self.v += (1.0 - self.b2) * (grad ** 2 - self.v)

        m = self.m / (1 - self.b1 ** self.t)
        v = self.v / (1 - self.b2 ** self.t)

        step = -self.lr * m / (v ** 0.5 + self.eps)
        return step


class Net:
    def __init__(self, layers):
        self.layers = layers
        self.phase = 'TRAIN'

    def __repr__(self):
        return '\n'.join([str(l) for l in self.layers])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        # back propagation
        layer_grads = []
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            layer_grads.append(copy.copy(layer.gradients))

        struct_grad = StructuredParam(layer_grads[::-1])
        struct_grad.wrt_input = grad
        return struct_grad

    @property
    def parameters(self):
        trainable = [l.parameters for l in self.layers]
        untrainable = [l.u_parameters for l in self.layers]
        return StructuredParam(trainable, untrainable)

    @parameters.setter
    def params(self, parameters):
        self.parameters.values = parameters.values
        self.parameters.ut_values = parameters.ut_values

    def get_phase(self):
        return self.phase

    def set_phase(self, phase):
        for layer in self.layers:
            layer.set_phase(phase)
        self.phase = phase

    def init_params(self, input_shape):
        self.forward(np.ones((1, *input_shape)))


class StructuredParam:
    def __init__(self, param_list, ut_param_list=None):
        self.param_list = param_list
        self.ut_param_list = ut_param_list

    @property
    def values(self):
        return np.array([v for p in self.param_list for v in p.values()])

    @values.setter
    def values(self, values):
        i = 0
        for d in self.param_list:
            for name in d.keys():
                d[name] = values[i]
                i += 1

    @property
    def ut_values(self):
        return np.array([v for p in self.ut_param_list for v in p.values()])

    @ut_values.setter
    def ut_values(self, values):
        i = 0
        for d in self.ut_param_list:
            for name in d.keys():
                d[name] = values[i]
                i += 1

    @property
    def shape(self):
        shape = list()
        for d in self.param_list:
            l_shape = dict()
            for k, v in d.items():
                l_shape[k] = v.shape
            shape.append(l_shape)
        shape = tuple(shape)
        return shape

    @staticmethod
    def _ensure_values(obj):
        if isinstance(obj, StructuredParam):
            obj = obj.values
        return obj

    def clip(self, min_=None, max_=None):
        obj = copy.deepcopy(self)
        obj.values = [v.clip(min_, max_) for v in self.values]
        return obj

    def __add__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values + self._ensure_values(other)
        return obj

    def __radd__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) + self.values
        return obj

    def __iadd__(self, other):
        self.values += self._ensure_values(other)
        return self

    def __sub__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values - self._ensure_values(other)
        return obj

    def __rsub__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) - self.values
        return obj

    def __isub__(self, other):
        other = self._ensure_values(other)
        self.values -= self._ensure_values(other)
        return self

    def __mul__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values * self._ensure_values(other)
        return obj

    def __rmul__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) * self.values
        return obj

    def __imul__(self, other):
        self.values *= self._ensure_values(other)
        return self

    def __truediv__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values / self._ensure_values(other)
        return obj

    def __rtruediv__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) / self.values
        return obj

    def __itruediv__(self, other):
        self.values /= self._ensure_values(other)
        return self

    def __pow__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self.values ** self._ensure_values(other)
        return obj

    def __ipow__(self, other):
        self.values **= self._ensure_values(other)
        return self

    def __neg__(self):
        obj = copy.deepcopy(self)
        obj.values = -self.values
        return obj

    def __len__(self):
        return len(self.values)

    def __lt__(self, other):
        obj = copy.deepcopy(self)
        other = self._ensure_values(other)

        if isinstance(other, float):
            obj.values = [v < other for v in self.values]
        else:
            obj.values = [v < other[i] for i, v in enumerate(self.values)]
        return obj

    def __gt__(self, other):
        obj = copy.deepcopy(self)
        other = self._ensure_values(other)

        if isinstance(other, float):
            obj.values = [v > other for v in self.values]
        else:
            obj.values = [v > other[i] for i, v in enumerate(self.values)]
        return obj

    def __and__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) & self.values
        return obj

    def __or__(self, other):
        obj = copy.deepcopy(self)
        obj.values = self._ensure_values(other) | self.values
        return obj


class Model:
    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.net.forward(inputs)

    def backward(self, preds, targets):
        loss = self.loss.loss(preds, targets)
        grad_from_loss = self.loss.grad(preds, targets)
        struct_grad = self.net.backward(grad_from_loss)
        return loss, struct_grad

    def apply_grads(self, grads):
        params = self.net.params
        self.optimizer.step(grads, params)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.net.params, f)

    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)

        self.net.params = params
        for layer in self.net.layers:
            layer.is_init = True

    def get_phase(self):
        return self.net.get_phase()

    def set_phase(self, phase):
        self.net.set_phase(phase)


class BaseIterator:
    def __call__(self, inputs, targets):
        raise NotImplementedError


class BatchIterator(BaseIterator):
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            idx = np.arange(len(inputs))
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start: end]
            batch_targets = targets[start: end]
            yield Batch(inputs=batch_inputs, targets=batch_targets)


def mean_square_error(predictions, targets):
    assert predictions.shape == targets.shape
    print(predictions.ndim)
    print(predictions.shape)
    if predictions.ndim == 1:
        mse = np.mean(np.square(predictions - targets))
    elif predictions.ndim == 2:
        mse = np.mean(np.sum(np.square(predictions - targets), axis=1))
    else:
        raise ValueError('predictions supposes to have 1 or 2 dim.')
    return {'mse': mse}


def read_data(data_dir='data'):
    path_A = '/'.join([data_dir, 'A.txt'])
    path_B = '/'.join([data_dir, 'B.txt'])
    path_M = '/'.join([data_dir, 'Mystery.txt'])
    A_str = open(path_A, 'r').read()
    B_str = open(path_B, 'r').read()
    M_str = open(path_M, 'r').read()
    A = A_str.split('\n\n')
    A.pop()
    for i in range(len(A)):
        A[i] = A[i].split('\n')
        for j in range(len(A[i])):
            A[i][j] = A[i][j].split(' ')
            A[i][j].pop()
            A[i][j] = list(map(lambda x: int(x), A[i][j]))
            A[i][j] = np.array(A[i][j])
    A = np.array(A)

    B = B_str.split('\n\n')
    B.pop()
    for i in range(len(B)):
        B[i] = B[i].split('\n')
        for j in range(len(B[i])):
            B[i][j] = B[i][j].split(' ')
            B[i][j].pop()
            B[i][j] = list(map(lambda x: int(x), B[i][j]))
            B[i][j] = np.array(B[i][j])
    B = np.array(B)

    M = M_str.split('\n\n')
    M.pop()
    for i in range(len(M)):
        M[i] = M[i].split('\n')
        for j in range(len(M[i])):
            M[i][j] = M[i][j].split(' ')
            M[i][j].pop()
            M[i][j] = list(map(lambda x: int(x), M[i][j]))
            M[i][j] = np.array(M[i][j])
    M = np.array(M)

    print(A.shape, B.shape, M.shape)
    train_x = np.zeros(shape=[A.shape[0] + B.shape[0], A.shape[1], A.shape[2]])
    train_x[0:A.shape[0]] = A[:]
    train_x[A.shape[0]:A.shape[0] + B.shape[0]] = B[:]
    train_y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    test_x = M
    return [train_x, train_y, test_x]


def print_pred(test_x, test_pred):
    test_pred = list(map(lambda x: 'B' if x[0] > 0.5 else 'A', test_pred))
    for i in range(test_x.shape[0]):
        print('test_data:')
        print(test_x[i])
        print('prediction class:', test_pred[i])
        print()


def simple_print_pred(test_x, test_pred):
    print(test_pred)
    test_pred = list(map(lambda x: 'B' if x[0] > 0.5 else 'A', test_pred))
    print(''.join(test_pred))


def main(model_type, lr, epochs, batch_size=10):
    train_x, train_y, test_x = read_data('data')
    origin_test_x = test_x.copy()

    if model_type == 'mlp':
        net = Net([
            Dense(5),
            Relu(),
            Dense(1),
            Sigmoid()
        ])
        train_x = train_x.reshape([-1, 25])
        train_y = train_y.reshape([-1, 1])
        test_x = test_x.reshape([-1, 25])
    elif model_type == 'cnn':
        train_x = train_x.reshape((-1, 5, 5, 1))
        train_y = train_y.reshape([-1, 1])
        test_x = test_x.reshape((-1, 5, 5, 1))
        net = Net([
            Conv2D(kernel=[3, 3, 1, 3], stride=[1, 1]),
            Relu(),
            MaxPool2D(pool_size=[2, 2], stride=[2, 2]),
            Reshape(-1),
            Dense(1),
            Sigmoid()
        ])
    else:
        raise ValueError('Invalid argument: model_type')

    model = Model(net=net, loss=L2(),
                  optimizer=Adam(lr=lr))

    iterator = BatchIterator(batch_size=batch_size)
    loss_list = list()
    for epoch in range(epochs):
        t_start = time.time()
        for batch in iterator(train_x, train_y):
            pred = model.forward(batch.inputs)
            loss, grads = model.backward(pred, batch.targets)
            model.apply_grads(grads)
            loss_list.append(loss)
        print('Epoch %d time cost: %.4f' % (epoch, time.time() - t_start), 'Epoch %d loss: %.8f' % (epoch, loss))
        # evaluation
        model.set_phase('TEST')
        test_pred = model.forward(test_x)
        # print_pred(origin_test_x, test_pred)
        simple_print_pred(origin_test_x, test_pred)
        model.set_phase('TRAIN')


if __name__ == '__main__':
    print('Perceptron:')
    main('mlp', 0.1, 30)
    print('')

    print('CNN:')
    main('cnn', 0.1, 30)
    print('')
