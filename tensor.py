"""
Custom autograd tensor implementation for neural network training.
Adapted from Gemini workspace with enhancements for game AI use case.
"""
import numpy as np


class Tensor:
    """
    A tensor with automatic differentiation support.

    This is a lightweight autograd implementation that tracks computation
    graphs and can compute gradients via backpropagation.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float64) if not isinstance(data, np.ndarray) else data.astype(np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad_output = out.grad
            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    grad_output = np.expand_dims(grad_output, axis)
                else:
                    for ax in sorted(axis):
                        grad_output = np.expand_dims(grad_output, ax)
            self.grad += np.broadcast_to(grad_output, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Compute mean with gradient support."""
        s = self.sum(axis=axis, keepdims=keepdims)
        n = self.data.size if axis is None else np.prod([self.data.shape[i] for i in ([axis] if isinstance(axis, int) else axis)])
        return s * (1.0 / n)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """Sigmoid activation with numerical stability."""
        sig = np.where(self.data >= 0,
                       1 / (1 + np.exp(-self.data)),
                       np.exp(self.data) / (1 + np.exp(self.data)))
        out = Tensor(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        """Tanh activation."""
        out = Tensor(np.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out

    def softmax(self, axis=-1):
        """Softmax activation with numerical stability."""
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_data = np.exp(shifted)
        sm = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(sm, (self,), 'softmax')

        def _backward():
            # Simplified softmax backward
            # For each sample, gradient is sm * (grad - sum(grad * sm))
            s = out.data
            grad_sum = np.sum(out.grad * s, axis=axis, keepdims=True)
            self.grad += s * (out.grad - grad_sum)
        out._backward = _backward

        return out

    def reshape(self, new_shape):
        out = Tensor(self.data.reshape(new_shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,), 'transpose')

        def _backward():
            inv_axes = np.argsort(axes)
            self.grad += out.grad.transpose(*inv_axes)
        out._backward = _backward
        return out

    def flatten(self):
        out = Tensor(self.data.flatten(), (self,), 'flatten')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def _unbroadcast(self, grad, shape):
        if grad.shape == shape:
            return grad

        ndims_added = grad.ndim - len(shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward(self):
        """Run backpropagation from this tensor."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def detach(self):
        """Create a copy without gradient tracking."""
        return Tensor(self.data.copy())


# ==================== IM2COL HELPERS ====================

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """Compute indices for im2col transformation."""
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """Transform input tensor to column format for efficient convolution."""
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=1, stride=1):
    """Transform column format back to tensor format."""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ==================== NEURAL NETWORK MODULES ====================

class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        """Zero out all parameter gradients."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        """Return list of trainable parameters."""
        return []

    def train(self):
        """Set module to training mode."""
        self._training = True
        return self

    def eval(self):
        """Set module to evaluation mode."""
        self._training = False
        return self


class Linear(Module):
    """Fully connected linear layer."""

    def __init__(self, nin, nout):
        scale = np.sqrt(2.0 / nin)  # He initialization
        self.w = Tensor(np.random.randn(nin, nout) * scale)
        self.b = Tensor(np.zeros(nout))

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class Conv2D(Module):
    """2D Convolutional layer using im2col."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.w = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.b = Tensor(np.zeros((out_channels,)))

    def __call__(self, x):
        # x: (N, C, H, W)
        # w: (F, C, HH, WW)

        FN, C, HH, WW = self.w.data.shape
        N, C_in, H, W = x.data.shape
        assert C == C_in, f"Input channels {C_in} must match filter channels {C}"

        H_out = (H + 2 * self.padding - HH) // self.stride + 1
        W_out = (W + 2 * self.padding - WW) // self.stride + 1

        # im2col
        x_cols = im2col_indices(x.data, HH, WW, self.padding, self.stride)
        w_col = self.w.data.reshape(FN, -1)

        # Convolution (GEMM)
        out_col = w_col @ x_cols + self.b.data.reshape(-1, 1)

        # Reshape to output
        out_data = out_col.reshape(FN, H_out, W_out, N).transpose(3, 0, 1, 2)

        out = Tensor(out_data, (x, self.w, self.b), 'Conv2D')

        def _backward():
            # dout: (N, F, H_out, W_out)
            dout = out.grad.transpose(1, 2, 3, 0).reshape(FN, -1)

            # db
            self.b.grad += np.sum(dout, axis=1)

            # dw
            dw_col = dout @ x_cols.T
            self.w.grad += dw_col.reshape(self.w.data.shape)

            # dx
            dx_col = w_col.T @ dout
            dx_data = col2im_indices(dx_col, x.data.shape, HH, WW, self.padding, self.stride)
            x.grad += dx_data

        out._backward = _backward
        return out

    def parameters(self):
        return [self.w, self.b]


class MaxPool2D(Module):
    """2D Max Pooling layer."""

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def __call__(self, x):
        N, C, H, W = x.data.shape
        HH, WW = self.kernel_size, self.kernel_size
        stride = self.stride

        H_out = (H - HH) // stride + 1
        W_out = (W - WW) // stride + 1

        # Reshape for max pooling
        x_reshaped = x.data.reshape(N, C, H_out, stride, W_out, stride)
        out_data = x_reshaped.max(axis=(3, 5))

        out = Tensor(out_data, (x,), 'MaxPool2D')

        def _backward():
            dout = out.grad

            # Create mask of max indices
            dx = np.zeros_like(x.data)

            x_reshaped_copy = x.data.reshape(N, C, H_out, stride, W_out, stride)
            out_broadcast = out_data.reshape(N, C, H_out, 1, W_out, 1)
            mask = (x_reshaped_copy == out_broadcast)

            dout_broadcast = dout.reshape(N, C, H_out, 1, W_out, 1)
            grad_reshaped = mask * dout_broadcast

            dx = grad_reshaped.reshape(N, C, H, W)
            x.grad += dx

        out._backward = _backward
        return out


class BatchNorm2D(Module):
    """2D Batch Normalization layer."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features))
        self.beta = Tensor(np.zeros(num_features))

        # Running statistics (not trainable)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self._training = True

    def __call__(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.data.shape

        if self._training:
            # Compute batch statistics
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x.data - mean.reshape(1, C, 1, 1)) / np.sqrt(var.reshape(1, C, 1, 1) + self.eps)

        # Scale and shift
        out_data = self.gamma.data.reshape(1, C, 1, 1) * x_norm + self.beta.data.reshape(1, C, 1, 1)

        out = Tensor(out_data, (x, self.gamma, self.beta), 'BatchNorm2D')

        def _backward():
            # Simplified backward for batch norm
            dout = out.grad

            # dgamma, dbeta
            self.gamma.grad += (dout * x_norm).sum(axis=(0, 2, 3))
            self.beta.grad += dout.sum(axis=(0, 2, 3))

            # dx (simplified)
            gamma_r = self.gamma.data.reshape(1, C, 1, 1)
            std_inv = 1.0 / np.sqrt(var.reshape(1, C, 1, 1) + self.eps)
            x.grad += gamma_r * std_inv * dout

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]


class Dropout(Module):
    """Dropout regularization layer."""

    def __init__(self, p=0.5):
        self.p = p
        self._training = True
        self._mask = None

    def __call__(self, x):
        if not self._training or self.p == 0:
            return x

        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float64)
        scale = 1.0 / (1.0 - self.p)
        out = Tensor(x.data * mask * scale, (x,), 'Dropout')
        self._mask = mask

        def _backward():
            x.grad += out.grad * mask * scale
        out._backward = _backward

        return out


# ==================== LOSS FUNCTIONS ====================

class MSELoss(Module):
    """Mean Squared Error loss."""

    def __call__(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true)
        return ((y_pred - y_true)**2).mean()


class CrossEntropyLoss(Module):
    """Cross Entropy loss with logits."""

    def __call__(self, logits, targets):
        # logits: (N, C)
        # targets: (N,) integer class indices
        N = logits.data.shape[0]

        # Softmax
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cross entropy
        targets_array = targets.data if isinstance(targets, Tensor) else np.array(targets)
        log_probs = np.log(probs[np.arange(N), targets_array.astype(int)] + 1e-10)
        loss = -np.mean(log_probs)

        out = Tensor(loss, (logits,), 'CrossEntropy')

        def _backward():
            grad = probs.copy()
            grad[np.arange(N), targets_array.astype(int)] -= 1
            logits.grad += grad / N
        out._backward = _backward

        return out


# ==================== OPTIMIZERS ====================

class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                p.data -= self.lr * self.velocities[i]
            else:
                p.data -= self.lr * grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class Adam:
    """Adam optimizer."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
