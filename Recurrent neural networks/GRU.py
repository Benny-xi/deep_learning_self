import argparse

import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l



"""初始化模型参数"""
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        # return nn.Parameter(torch.randn(shape).to(device)*0.01)
        return torch.randn(size=shape, device=device) * 0.01
    def three():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            # torch.zeros(num_hiddens).to(device)
            torch.zeros(num_hiddens, device=device)
        )
    W_xz, W_hz, b_z = three()   # 更新门参数
    W_xr, W_hr, b_r = three()   # 重置门参数
    W_xh, W_hh, b_h = three()   # 候选隐藏状态参数
    W_hq, b_q = normal((num_hiddens, num_outputs)), torch.zeros(num_outputs).to(device)     # 输出层参数

    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


"""定义隐状态初始化函数"""
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)

"""定义门控线性单元（GRU）"""
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for i in range(len(inputs)):
        Z = torch.sigmoid((inputs[i] @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((inputs[i] @ W_xr) + (H @ W_hr) + b_r)
        C = torch.tanh((inputs[i] @ W_xh) + ((R * H) @ W_hh) + b_h)
        H_out = Z * H + (1 - Z) * C
        Y = (H_out @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H_out,)

"""书上的 GRU"""
def gru_d2l(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


"""训练模型"""
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):


    loss = nn.CrossEntropyLoss()
    ppl_history = []
    epoches = []

    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: d2l.predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = d2l.train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            print(f"Epoch {epoch + 1}, Perplexity: {ppl}")  # 打印出 epoch 和 ppl 的值
            ppl_history.append(float(ppl))
            epoches.append(epoch + 1)

    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    return ppl_history, epoches

def train_ch8_d2l(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):


    loss = nn.CrossEntropyLoss()
    ppl_history = []
    epoches = []

    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: d2l.predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = d2l.train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            print(f"Epoch {epoch + 1}, Perplexity: {ppl}")  # 打印出 epoch 和 ppl 的值
            ppl_history.append(float(ppl))
            epoches.append(epoch + 1)

    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    return ppl_history, epoches

# vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
# num_epochs, lr = 500, 1e-3
# model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
# train_ch8(model, train_iter, vocab, lr, num_epochs, device)

if __name__ == '__main__':

    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model_d2l = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru_d2l)
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
    perplexity_h, epochs_h = train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    perplexity_h_d2l, epochs_h_d2l = train_ch8_d2l(model_d2l, train_iter, vocab, lr, num_epochs, device)

    plt.xlabel(xlabel='epoch')
    plt.ylabel(ylabel='perplexity')
    plt.plot(perplexity_h, label='xph', )
    plt.plot(perplexity_h_d2l, label='d2l', )

    plt.legend()
    plt.grid(True)
    plt.show()


