import torch 

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y_pred = w * x
y_true = torch.tensor(10.0)

loss1 = (w * x - y_true) ** 2
loss1.backward()
print("After Ist backward",w.grad.item())

w.grad.zero_()

loss2 = (w * x - y_true) ** 2
loss2.backward()
print("After Ist backward",w.grad.item())

