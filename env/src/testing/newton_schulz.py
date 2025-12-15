# According to Gemini 3,
# X_{k+1} = 0.5 X_{k} (3I - X_{k}^TX_{K})

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

A = torch.randn(4, 4)

X = A / ((A**2.0).sum() ** 0.5)

a, b, c = (3.4445, -4.7750, 2.0315)

for k in range(5):
    X = a * X + b * (X @ X.T @ X) + c * (X @ X.T @ X @ X.T) @ X
print(X)
print(X.T @ X)
print(X @ X.T)
plt.imshow(X.T @ X)
plt.show()