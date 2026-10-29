"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
def row_max(matrix):
    return np.max(matrix, axis=1, keepdims=True)

# Step 3 - row_sum
def row_sum(matrix):
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
def exp_shifted(logits):
    return np.exp(logits - row_max(logits))

# Step 5 - stable_softmax
def stable_softmax(logits):
    shifted = exp_shifted(logits)
    return shifted / row_sum(shifted)

# Step 6 - one_hot
def one_hot(labels, num_classes):
    labels = np.asarray(labels).astype(int)
    out = np.zeros((labels.shape[0], num_classes), dtype=float)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    return probs[np.arange(probs.shape[0]), labels]

# Step 8 - cross_entropy_loss
def cross_entropy_loss(probs, labels, eps=1e-12):
    true_probs = gather_true_class_probs(probs, labels)
    return float(-np.mean(np.log(np.clip(true_probs, eps, 1.0))))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    preds = argmax_rows(logits_or_probs)
    return float(np.mean(preds == labels))

# Step 10 - he_std
def he_std(fan_in):
    return float(np.sqrt(2.0 / fan_in))

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    std = he_std(fan_in)
    return np.random.normal(0.0, std, size=shape).astype(np.float64)

# Step 12 - init_zero_bias
def init_zero_bias(length):
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    if pad == 0:
        return images
    return np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    return (input_size + 2 * padding - kernel) // stride + 1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    N, C, H, W = images.shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    padded = pad_2d(images, padding)
    
    cols = np.zeros((N * out_h * out_w, C * kernel_h * kernel_w), dtype=images.dtype)
    row_idx = 0
    for n in range(N):
        for i in range(out_h):
            h_start = i * stride
            for j in range(out_w):
                w_start = j * stride
                patch = padded[n, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w]
                cols[row_idx] = patch.reshape(-1)
                row_idx += 1
    return cols

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=np.float64)
    
    row_idx = 0
    for n in range(N):
        for i in range(out_h):
            h_start = i * stride
            for j in range(out_w):
                w_start = j * stride
                patch = cols[row_idx].reshape(C, kernel_h, kernel_w)
                padded[n, :, h_start:h_start+kernel_h, w_start:w_start+kernel_w] += patch
                row_idx += 1
                
    if padding > 0:
        return padded[:, :, padding:padding+H, padding:padding+W]
    return padded

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weights.shape
    out_h = output_spatial_size(H, kH, stride, padding)
    out_w = output_spatial_size(W, kW, stride, padding)
    cols = im2col(x, kH, kW, stride, padding)
    W_flat = weights.reshape(C_out, -1)
    out = cols @ W_flat.T + bias.reshape(1, -1)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    cache = {'x_shape': x.shape, 'weights': weights, 'cols': cols,
             'stride': stride, 'padding': padding, 'kernel_h': kH, 'kernel_w': kW}
    return out, cache

def conv2d_grad_input(d_out, cache):
    W_row = cache['weights'].reshape(cache['weights'].shape[0], -1)
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    d_cols = d_out_row @ W_row
    return col2im(d_cols, cache['x_shape'], cache['kernel_h'], cache['kernel_w'],
                  cache['stride'], cache['padding'])

def conv2d_grad_weights(d_out, cache):
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    return (d_out_row.T @ cache['cols']).reshape(cache['weights'].shape)

def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))

def conv2d_backward(d_out, cache):
    return conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out)

# Step 18 - conv2d_grad_input
def conv2d_forward(x, weights, bias, stride, padding):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weights.shape
    out_h = output_spatial_size(H, kH, stride, padding)
    out_w = output_spatial_size(W, kW, stride, padding)
    cols = im2col(x, kH, kW, stride, padding)
    W_flat = weights.reshape(C_out, -1)
    out = cols @ W_flat.T + bias.reshape(1, -1)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    cache = {'x_shape': x.shape, 'weights': weights, 'cols': cols,
             'stride': stride, 'padding': padding, 'kernel_h': kH, 'kernel_w': kW}
    return out, cache

def conv2d_grad_input(d_out, cache):
    W_row = cache['weights'].reshape(cache['weights'].shape[0], -1)
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    d_cols = d_out_row @ W_row
    return col2im(d_cols, cache['x_shape'], cache['kernel_h'], cache['kernel_w'],
                  cache['stride'], cache['padding'])

def conv2d_grad_weights(d_out, cache):
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    return (d_out_row.T @ cache['cols']).reshape(cache['weights'].shape)

def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))

def conv2d_backward(d_out, cache):
    return conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out)

# Step 19 - conv2d_grad_weights
def conv2d_forward(x, weights, bias, stride, padding):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weights.shape
    out_h = output_spatial_size(H, kH, stride, padding)
    out_w = output_spatial_size(W, kW, stride, padding)
    cols = im2col(x, kH, kW, stride, padding)
    W_flat = weights.reshape(C_out, -1)
    out = cols @ W_flat.T + bias.reshape(1, -1)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    cache = {'x_shape': x.shape, 'weights': weights, 'cols': cols,
             'stride': stride, 'padding': padding, 'kernel_h': kH, 'kernel_w': kW}
    return out, cache

def conv2d_grad_input(d_out, cache):
    W_row = cache['weights'].reshape(cache['weights'].shape[0], -1)
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    d_cols = d_out_row @ W_row
    return col2im(d_cols, cache['x_shape'], cache['kernel_h'], cache['kernel_w'],
                  cache['stride'], cache['padding'])

def conv2d_grad_weights(d_out, cache):
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    return (d_out_row.T @ cache['cols']).reshape(cache['weights'].shape)

def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))

def conv2d_backward(d_out, cache):
    return conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out)

# Step 20 - conv2d_grad_bias
def conv2d_forward(x, weights, bias, stride, padding):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weights.shape
    out_h = output_spatial_size(H, kH, stride, padding)
    out_w = output_spatial_size(W, kW, stride, padding)
    cols = im2col(x, kH, kW, stride, padding)
    W_flat = weights.reshape(C_out, -1)
    out = cols @ W_flat.T + bias.reshape(1, -1)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    cache = {'x_shape': x.shape, 'weights': weights, 'cols': cols,
             'stride': stride, 'padding': padding, 'kernel_h': kH, 'kernel_w': kW}
    return out, cache

def conv2d_grad_input(d_out, cache):
    W_row = cache['weights'].reshape(cache['weights'].shape[0], -1)
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    d_cols = d_out_row @ W_row
    return col2im(d_cols, cache['x_shape'], cache['kernel_h'], cache['kernel_w'],
                  cache['stride'], cache['padding'])

def conv2d_grad_weights(d_out, cache):
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    return (d_out_row.T @ cache['cols']).reshape(cache['weights'].shape)

def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))

def conv2d_backward(d_out, cache):
    return conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out)

# Step 21 - conv2d_backward
def conv2d_forward(x, weights, bias, stride, padding):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weights.shape
    out_h = output_spatial_size(H, kH, stride, padding)
    out_w = output_spatial_size(W, kW, stride, padding)
    cols = im2col(x, kH, kW, stride, padding)
    W_flat = weights.reshape(C_out, -1)
    out = cols @ W_flat.T + bias.reshape(1, -1)
    out = out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    cache = {'x_shape': x.shape, 'weights': weights, 'cols': cols,
             'stride': stride, 'padding': padding, 'kernel_h': kH, 'kernel_w': kW}
    return out, cache

def conv2d_grad_input(d_out, cache):
    W_row = cache['weights'].reshape(cache['weights'].shape[0], -1)
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    d_cols = d_out_row @ W_row
    return col2im(d_cols, cache['x_shape'], cache['kernel_h'], cache['kernel_w'],
                  cache['stride'], cache['padding'])

def conv2d_grad_weights(d_out, cache):
    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, cache['weights'].shape[0])
    return (d_out_row.T @ cache['cols']).reshape(cache['weights'].shape)

def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))

def conv2d_backward(d_out, cache):
    return conv2d_grad_input(d_out, cache), conv2d_grad_weights(d_out, cache), conv2d_grad_bias(d_out)

# Step 22 - maxpool2d_forward (not yet solved)
# TODO: implement

# Step 23 - scatter_grad_window
def maxpool2d_forward(x, kernel, stride):
    N, C, H, W = x.shape
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=int)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    window = x[n, c, i*stride:i*stride+kernel, j*stride:j*stride+kernel]
                    out[n, c, i, j] = np.max(window)
                    argmax[n, c, i, j] = np.argmax(window)
    return out, {'x_shape': x.shape, 'argmax': argmax, 'kernel': kernel, 'stride': stride}

def scatter_grad_window(grad_value, argmax_index, kernel):
    w = np.zeros((kernel, kernel), dtype=float)
    w[argmax_index // kernel, argmax_index % kernel] = grad_value
    return w

def maxpool2d_backward(d_out, cache):
    N, C, H, W = cache['x_shape']
    dx = np.zeros((N, C, H, W), dtype=float)
    for n in range(N):
        for c in range(C):
            for i in range(d_out.shape[2]):
                for j in range(d_out.shape[3]):
                    dx[n, c, i*cache['stride']:i*cache['stride']+cache['kernel'],
                             j*cache['stride']:j*cache['stride']+cache['kernel']] += \
                        scatter_grad_window(d_out[n, c, i, j], cache['argmax'][n, c, i, j], cache['kernel'])
    return dx

# Step 24 - maxpool2d_backward
def maxpool2d_forward(x, kernel, stride):
    N, C, H, W = x.shape
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=int)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    window = x[n, c, i*stride:i*stride+kernel, j*stride:j*stride+kernel]
                    out[n, c, i, j] = np.max(window)
                    argmax[n, c, i, j] = np.argmax(window)
    return out, {'x_shape': x.shape, 'argmax': argmax, 'kernel': kernel, 'stride': stride}

def scatter_grad_window(grad_value, argmax_index, kernel):
    w = np.zeros((kernel, kernel), dtype=float)
    w[argmax_index // kernel, argmax_index % kernel] = grad_value
    return w

def maxpool2d_backward(d_out, cache):
    N, C, H, W = cache['x_shape']
    dx = np.zeros((N, C, H, W), dtype=float)
    for n in range(N):
        for c in range(C):
            for i in range(d_out.shape[2]):
                for j in range(d_out.shape[3]):
                    dx[n, c, i*cache['stride']:i*cache['stride']+cache['kernel'],
                             j*cache['stride']:j*cache['stride']+cache['kernel']] += \
                        scatter_grad_window(d_out[n, c, i, j], cache['argmax'][n, c, i, j], cache['kernel'])
    return dx

# Step 25 - relu_forward
def relu_forward(x):
    return np.maximum(0, x), {'x': x}

def relu_backward(d_out, cache):
    return d_out * (cache['x'] > 0)

# Step 26 - relu_backward
def relu_forward(x):
    return np.maximum(0, x), {'x': x}

def relu_backward(d_out, cache):
    return d_out * (cache['x'] > 0)

# Step 27 - flatten_forward
def flatten_forward(x):
    return x.reshape(x.shape[0], -1), {'x_shape': x.shape}

def flatten_backward(d_out, cache):
    return d_out.reshape(cache['x_shape'])

# Step 28 - flatten_backward
def flatten_forward(x):
    return x.reshape(x.shape[0], -1), {'x_shape': x.shape}

def flatten_backward(d_out, cache):
    return d_out.reshape(cache['x_shape'])

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    return x @ weights + bias, {'x': x, 'weights': weights}

def linear_grad_input(d_out, cache):
    return d_out @ cache['weights'].T

def linear_grad_weights(x, dout):
    return x.T @ dout

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)

def linear_backward(dout, cache):
    return linear_grad_input(dout, cache), linear_grad_weights(cache['x'], dout), linear_grad_bias(dout)

# Step 30 - linear_grad_input
def linear_forward(x, weights, bias):
    return x @ weights + bias, {'x': x, 'weights': weights}

def linear_grad_input(d_out, cache):
    return d_out @ cache['weights'].T

def linear_grad_weights(x, dout):
    return x.T @ dout

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)

def linear_backward(dout, cache):
    return linear_grad_input(dout, cache), linear_grad_weights(cache['x'], dout), linear_grad_bias(dout)

# Step 31 - linear_grad_weights
def linear_forward(x, weights, bias):
    return x @ weights + bias, {'x': x, 'weights': weights}

def linear_grad_input(d_out, cache):
    return d_out @ cache['weights'].T

def linear_grad_weights(x, dout):
    return x.T @ dout

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)

def linear_backward(dout, cache):
    return linear_grad_input(dout, cache), linear_grad_weights(cache['x'], dout), linear_grad_bias(dout)

# Step 32 - linear_grad_bias
def linear_forward(x, weights, bias):
    return x @ weights + bias, {'x': x, 'weights': weights}

def linear_grad_input(d_out, cache):
    return d_out @ cache['weights'].T

def linear_grad_weights(x, dout):
    return x.T @ dout

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)

def linear_backward(dout, cache):
    return linear_grad_input(dout, cache), linear_grad_weights(cache['x'], dout), linear_grad_bias(dout)

# Step 33 - linear_backward
def linear_forward(x, weights, bias):
    return x @ weights + bias, {'x': x, 'weights': weights}

def linear_grad_input(d_out, cache):
    return d_out @ cache['weights'].T

def linear_grad_weights(x, dout):
    return x.T @ dout

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)

def linear_backward(dout, cache):
    return linear_grad_input(dout, cache), linear_grad_weights(cache['x'], dout), linear_grad_bias(dout)

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

