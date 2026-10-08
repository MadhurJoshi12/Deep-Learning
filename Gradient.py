import torch
x = torch.tensor(2.0)

w = torch.tensor(3.0, requires_grad=True)

y_true = torch.tensor(10.0)

lr = 0.1

y_pred = w*x
loss = (y_pred - y_true) ** 2

loss.backward()
# .item() converts a tensor containing a single value into a normal Python number (float or int).
print("Before update")
print("w: ", w.item())
print("Loss: ", loss.item())
print("Loss: ", w.grad.item())

with torch.no_grad(): # do track this
    w = w - lr * w.grad

w.grad.zero()
print("After update")
print("w", w.item())
