import torch

@torch.no_grad
def train(w: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, num_steps: int = 100, lr: float = 0.1) -> None:
    for step in range(num_steps):
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            
            y_pred = x @ w
            
            error = (y - y_pred)
            w += lr  * error * x
      
      
dim = 3
n_samples = 100  

w = torch.zeros(dim)

xs = torch.randn(n_samples, dim)

w_ref = torch.randn(dim) / (dim ** 0.5)

ys = xs @ w_ref + torch.randn(n_samples) * 0.05

print("pre-train MSE: ", ((xs @ w - ys)**2).mean())

train(w, xs, ys)

print("MSE: ", ((xs @ w - ys)**2).mean())

print(w)

print()

print(w_ref)

# seems like error must be 1-dimensional

# if do gradient descent

# l(y, y_hat) = y - y_hat
# dl/dy_hat = - 1
# l(y_hat(w)) = y - (x @ w)
# dl/dw = dl/dy_hat * dy_hat/dw = -1 * x

x = torch.randn(dim, requires_grad=True)
w = torch.randn(dim, requires_grad=True)
y = torch.randn(1, requires_grad=True)

loss = y - (x @ w)
loss.backward()

print()
# same
print(w.grad) 
print(-x)

# update:
# w -= grad * lr
# w += x * lr

# why does delta rule add in error? is it faster?
x = torch.randn(n_samples, dim)
w = torch.randn(dim, requires_grad=True)
with torch.no_grad():
    w /=  (dim**0.5)
w_ref = torch.randn(dim) / (dim ** 0.5)
y = x @ w_ref + torch.randn(n_samples) * 0.05

print("pre-train MSE: ", ((x @ w - y)**2).mean())

for step in range(100):
    
    loss = (y - (x @ w)).abs().mean()
    
    w.grad = None
    loss.backward()
    
    with torch.no_grad():
        w -= 0.1 * w.grad

print("MSE: ", ((x @ w - y)**2).mean())

# hmm ig slightly more complex with absolute values and all
# maybe delta rule is just a simplification