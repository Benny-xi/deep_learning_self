import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
d2l.plt.show()

# # Convert torch tensors to numpy arrays
# x_np = x.detach().numpy()
# y_np = y.detach().numpy()
# grad_np = x.grad.numpy()
#
# # Plotting
# plt.figure(figsize=(4.5, 2.5))
# plt.plot(x_np, y_np, label='sigmoid')
# plt.plot(x_np, grad_np, label='gradient')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

