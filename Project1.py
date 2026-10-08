import torch

# Input x value 
x = torch.tensor([1.,2.,3.,4.])

# y value
y = torch.tensor([2.,4.,6.,8.])

# Wrong guesse
w = torch.tensor([0.0])
b = torch.tensor([0.0])

# Predicton

y_pred = w * x + b

loss = ((y_pred - y) ** 2).mean()


print("Predicted Values", y_pred)
print("Actual Value", y)
print("Loss", loss)
