import torch
x = torch.tensor([1.,2.,3.,4.])
y_true = torch.tensor([2.,4.,6.,8.])    #Correct numbers

w = torch.tensor(0.0, requires_grad=True)

lr = 0.1
epochs = 10     #How many times training loop will run

for epoch in range(epochs):
    # Prediction
    y_pred = w * x

    # Loss
    loss = ((y_pred - y_true) **2).mean() 
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad

    w.grad.zero_()
    print(f"Epoch {epoch + 1}: w={w.item():.4f}, loss = {loss.item():.4f}")
    