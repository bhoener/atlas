import torch
import matplotlib.pyplot as plt
from linear_transformer import causal_polysketch_attention, poly_sketch_non_negative

b, k, m, r, p = (4, 8, 4, 6, 4)
B, L, H, h = (4, 128, 8, 4)
D = H * h

Q, K, V = torch.randn(3, B, L, D)

output = causal_polysketch_attention(Q, K, V, h, r, p)

out2 = poly_sketch_non_negative(Q, r, p)
print(out2.max(), out2.std(), out2.mean(), out2.min())
print(output.max(), output.std(), output.mean())

x = torch.randn(8)
x = torch.kron(x, x)
y = torch.randn(8)
y = torch.kron(y, y)

out = torch.dot(x, y)
print(out)

xs = []
ys = []
for test_r in range(2, 64, 2):
    xs.append(test_r)
    ys.append(poly_sketch_non_negative(Q, test_r, p).std())

plt.plot(xs, ys)
plt.show()