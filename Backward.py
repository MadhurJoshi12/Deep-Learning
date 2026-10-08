import torch 

w = torch.tensor(2.0, requires_grad=True)
# w is learnable value
x = torch.tensor(3.0)
y_pred = w * x
y_true = torch.tensor(10.0)

loss = (y_pred - y_true) ** 2
print(loss)
loss.backward()
print(w.grad)
'''
If w.grad is negative(<0) so increase it.
If w.grad is positive(>0) so decrease it.
'''

print("Prediction", y_pred.item())
print("loss", loss.item())  #loss = (w*x - 10)²
print("w.grad", w.grad.item())
'''
d(loss)/dw = 2*(w*x - 10) * x

Plug values:

= 2*(6 - 10) * 3
= 2*(-4)*3
w.grad = -24
'''