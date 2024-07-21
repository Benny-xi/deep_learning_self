import torch
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l



"""训练模型"""
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):


    loss = nn.CrossEntropyLoss()
    ppl_history = []


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

    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    return ppl_history


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, num_layers = len(vocab), 256, 3
    num_inputs = vocab_size
    device = d2l.try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)

    num_epochs, lr = 500, 2
    perplexity_h=train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    # plt.figure(figsize=(6, 8))
    # plt.plot(range(num_epochs), perplexity_h, marker='o')
    plt.plot(perplexity_h, marker='o', label='perplexity', color='blue')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()