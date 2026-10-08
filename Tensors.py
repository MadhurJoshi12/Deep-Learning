import torch
import numpy as np

One_Dimension = torch.tensor([1,2,3,4,5])
print(One_Dimension)

Second_Dimension = torch.tensor([
    [1,2],
    [3,4]
])
print(Second_Dimension)

zeros_tensor = torch.zeros(3,2)
print(zeros_tensor)

ones_tensor = torch.ones(3,2)
print(ones_tensor)

random_tensor = torch.rand(3,2) # Random decimal number
print(random_tensor)


arr = np.array([1,2,3,4,5])
n = torch.from_numpy(arr)
# Here arr and n are pointing same memory