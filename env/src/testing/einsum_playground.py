import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)

# dot product

a = torch.randn(8)
b = torch.randn(8)

c = torch.einsum("j, j ->", a, b)

print(c)


# matmul

A = torch.randn(8, 4)
B = torch.randn(4, 12)

C = torch.einsum("ij, jk -> ik", A, B)
print(C.shape)


# batched matmul

A = torch.randn(4, 8, 4)
B = torch.randn(4, 4, 12)

C = torch.einsum("bij, bjk -> bik", A, B)
print(C.shape)

# tensor contraction (matmul of any two dims)

A = torch.randn(3, 8, 2, 7, 9, 5)
B = torch.randn(5, 2, 4, 6, 3, 1)

C = torch.einsum("abcdef, hcijkl -> adefhjkl", A, B)
print(C.shape)


B, P, Q, R = (128, 256, 512, 256)
t0 = time.time()
einsum_result = torch.einsum(
    "bpq, bqr -> bpr", torch.randn(B, P, Q), torch.randn(B, Q, R)
)
torch.cuda.synchronize()
print(f"Einsum: {time.time() - t0}")

t0 = time.time()
mamtul_result = torch.randn(B, P, Q) @ torch.randn(B, Q, R)
torch.cuda.synchronize()
print(f"Matmul: {time.time() - t0}")


a = torch.randn(3, 8)
b = torch.randn(8, 2)

c = torch.einsum("ij, jk -> ik", a, b)
print(c.shape)


# transpose

a = torch.randn(8, 4)

a = torch.einsum("ij -> ji", a)

print(a.shape)

# multiple multiplications

A = torch.randn(2, 4, 8)
B = torch.randn(2, 8, 12)
C = torch.randn(2, 12, 6)

D = torch.einsum("bij, bjk, ckl -> bil", A, B, C)

print(D.shape, (A @ B @ C).shape)
