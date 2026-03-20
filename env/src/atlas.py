import torch
import torch.nn as nn
import torch.nn.functional as F
import train
import math
from dataloader import DataLoader
from linear_transformer import poly_sketch_non_negative


def newtonschulz5(
    x: torch.Tensor,
    iterations: int = 5,
    coeffs: tuple[float] = (3.4445, -4.775, 2.0315),
    epsilon: float = 1e-5,
) -> torch.Tensor:
    x = x / (x.norm() + epsilon)

    for _ in range(iterations):
        x = (
            coeffs[0] * x
            + coeffs[1] * x @ x.T @ x
            + coeffs[2] * (x @ x.T) @ (x @ x.T) @ x
        )

    return x


class Memory(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p: int = 2,
        chunk_size: int = 4,
        alpha: float = 0.9,
        theta: float = 0.95,
    ):
        """
        ATLAS memory module

        d_model (int): the dimension of the weights Wq, Wk, Wv
        n_heads (int): number of heads to split the input dimensions into
        p (int): degree of polynomial mapping
        chunk_size (int): how many tokens to calculate gradient for in parallel
        alpha (float): decay for M_t
        theta (float): decay for S_t
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.p = p
        self.chunk_size = chunk_size

        self.alpha = alpha
        self.theta = theta

        assert d_model % n_heads == 0

        self.d_key = d_model // n_heads

        self.register_buffer(
            "mask", torch.tril(torch.ones(self.chunk_size, self.chunk_size))
        )

    def forward_chunk(
        self,
        state: tuple[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_batch: bool = True,
    ) -> tuple[torch.Tensor]:
        B, H, L, d = q.size()
        (M_w1, M_w2, S_w1, S_w2) = state
        preact = k @ M_w1  # (L, d) x (d, 4d) -> (L, 4d)

        l1 = F.sigmoid(preact)

        out = l1 @ M_w2

        diff = out - v  # (L, d_k)

        # loss = ((out - v) ** 2.0).mean()

        ddiff = (
            1 / diff.numel() * torch.ones_like(diff, device=q.device)
        )  # (B, H, L, d_k)

        if is_batch:
            dM_w2 = torch.einsum("bhdi, bhik -> bhidk", l1.transpose(-2, -1), ddiff)
            dM_w2 = torch.einsum("bhidk, ij -> dk", dM_w2, self.mask)
            dl1 = ddiff @ M_w2.T  # (B, H, L, d_k) x (d_k, 4d) -> (B, H, L, 4d)
            dl1preact = (
                dl1 * l1 * (1 - l1)
            )  # (B, H, L, 4d) * (B, H, L, 4d) -> (B, H, L, 4d)
            dM_w1 = torch.einsum("bhdi, bhif -> bhidf", k.transpose(-2, -1), dl1preact)
            dM_w1 = torch.einsum("bhidf, ij -> df", dM_w1, self.mask)
        else:
            dM_w2 = torch.einsum("bhdi, bhik -> dk", l1.transpose(-2, -1), ddiff)
            dl1 = ddiff @ M_w2.T  # (B, H, L, d_k) x (d_k, 4d) -> (B, H, L, 4d)
            dl1preact = (
                dl1 * l1 * (1 - l1)
            )  # (B, H, L, 4d) * (B, H, L, 4d) -> (B, H, L, 4d)
            dM_w1 = torch.einsum("bhdi, bhif -> df", k.transpose(-2, -1), dl1preact)

        with torch.no_grad():
            S_w1 = (self.theta**self.chunk_size) * S_w1 - torch.diag(
                torch.ones(dM_w1.size(0), device=q.device) * self.theta
            ) @ torch.diag(
                torch.ones(dM_w1.size(0), device=q.device) * self.alpha
            ) @ dM_w1
            S_w2 = (self.theta**self.chunk_size) * S_w2 - torch.diag(
                torch.ones(dM_w2.size(0), device=q.device) * self.theta
            ) @ torch.diag(
                torch.ones(dM_w2.size(0), device=q.device) * self.alpha
            ) @ dM_w2
            S_w1 = newtonschulz5(S_w1)
            S_w2 = newtonschulz5(S_w2)
            M_w1 = self.alpha * M_w1 + S_w1
            M_w2 = self.alpha * M_w2 + S_w2

        output = F.sigmoid(q @ M_w1) @ M_w2
        return (M_w1, M_w2, S_w1, S_w2), output

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        state: tuple[torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        B, H, L, d = Q.size()
        if state is None:
            M_w1 = torch.randn(d, d * 4, device=Q.device)
            M_w2 = torch.randn(d * 4, self.d_key, device=Q.device)
            nn.init.orthogonal_(M_w1, 1 / (d**0.5))
            nn.init.orthogonal_(M_w2, 1 / ((d * 4) ** 0.5))

            S_w1 = torch.zeros_like(M_w1, device=Q.device)
            S_w2 = torch.zeros_like(M_w2, device=Q.device)

            output = torch.zeros_like(V)
            for chunk in range(1, L // self.chunk_size + 1):
                idx = torch.arange(
                    (chunk - 1) * self.chunk_size,
                    chunk * self.chunk_size,
                    device=Q.device,
                )
                q = Q[:, :, idx]
                k = K[:, :, idx]
                v = V[:, :, idx]

                (M_w1, M_w2, S_w1, S_w2), out = self.forward_chunk(
                    (M_w1, M_w2, S_w1, S_w2), q, k, v
                )
                output[:, :, idx] = out
            return output
        else:
            return self.forward_chunk(state, Q, K, V, is_batch=False)


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        self.__padding = kernel_size - 1
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.__padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[..., : -self.__padding]


class AtlasLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, r: int = 16, p: int = 2, chunk_size: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.d_key = d_model // n_heads

        self.r = r
        self.p = p

        self.mem = Memory(d_model=d_model, n_heads=n_heads, p=p, chunk_size=chunk_size)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wres = nn.Linear(d_model, d_model)

        self.convq = CausalConv1d(d_model, d_model, kernel_size=4)
        self.convk = CausalConv1d(d_model, d_model, kernel_size=4)

        self.convv = CausalConv1d(d_model, d_model, kernel_size=4)

        self.qknorm = nn.LayerNorm(d_model)
        self.postnorm = nn.LayerNorm(d_model)

        self.act = nn.SiLU()

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor] | None = None
    ) -> torch.Tensor:
        B, L, D = x.size()
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        res = self.act(self.wres(x))

        Q = self.act(self.convq(Q.transpose(1, 2))).transpose(1, 2)

        K = self.act(self.convk(K.transpose(1, 2))).transpose(1, 2)

        V = self.act(self.convv(V.transpose(1, 2))).transpose(1, 2)

        Q = self.qknorm(Q)
        K = self.qknorm(K)

        Q = Q.view(B, L, self.n_heads, self.d_key).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.d_key).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_key).transpose(1, 2)

        Q = poly_sketch_non_negative(Q, self.r, self.p)
        K = poly_sketch_non_negative(K, self.r, self.p)

        if state is not None:
            state, mem_out = self.mem(Q, K, V, state)
            mem_out = mem_out.transpose(1, 2).reshape(B, L, D)
            out = self.postnorm(mem_out)
            return state, out * res
        else:
            mem_out = self.mem(Q, K, V).transpose(1, 2).view(B, L, D)
            out = self.postnorm(mem_out)
            return out * res


class SwiGLU(nn.Module):
    def __init__(self, in_size: int, out_size: int, beta: float = 1.0):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.beta = beta

        self.W = nn.Linear(in_size, out_size)
        self.V = nn.Linear(in_size, out_size)

        self.sigmoid = nn.Sigmoid()

    def swish(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.beta * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.W(x)) * self.V(x)


class AtlasBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        r: int = 16,
        p: int = 2,
        chunk_size: int = 4,
        swiglu_beta: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.r = r
        self.p = p
        self.chunk_size = chunk_size

        self.swiglu_beta = swiglu_beta

        self.atlas_layer = AtlasLayer(
            d_model=d_model, n_heads=n_heads, r=r, p=p, chunk_size=chunk_size
        )

        self.pre_rms = nn.RMSNorm(d_model)

        self.post_rms = nn.RMSNorm(d_model)

        self.swiglu = SwiGLU(in_size=d_model, out_size=d_model, beta=swiglu_beta)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor] | None = None):
        if state is not None:
            state, x_res = self.atlas_layer(self.pre_rms(x), state)
            x = x + x_res
            x = x + self.swiglu(self.post_rms(x))
            return state, x
        else:
            x = x + self.atlas_layer(self.pre_rms(x))
            x = x + self.swiglu(self.post_rms(x))
            return x


class Atlas(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        seq_len: int = 1024,
        r: int = 8,
        p: int = 2,
        chunk_size: int = 4,
        swiglu_beta: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.r = r
        self.p = p
        self.chunk_size = chunk_size
        self.swiglu_beta = swiglu_beta

        self.emb = nn.Embedding(self.vocab_size, self.d_model)

        self.pos_emb = nn.Embedding(self.seq_len, self.d_model)

        self.layers = nn.ModuleList(
            AtlasBlock(
                d_model=d_model,
                n_heads=n_heads,
                p=p,
                chunk_size=chunk_size,
                swiglu_beta=swiglu_beta,
            )
            for _ in range(self.n_layers)
        )

        self.out_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.02)

    @property
    def poly_dim(self):
        return poly_sketch_non_negative(torch.randn(1, 1, 1, self.d_model // self.n_heads), self.r, self.p).size(-1)

    def forward(
        self, x: torch.Tensor, states: list[tuple[torch.Tensor]] | None = None
    ) -> torch.Tensor:
        B, L = x.size()

        x = self.emb(x)

        x = x + self.pos_emb(torch.arange(0, L).to(x.device))
        
        if states is not None:
            for i, layer in enumerate(self.layers):
                states[i], x = layer(x, states[i])

            return states, self.out_proj(x)
        else:
            for layer in self.layers:
                x = layer(x)

            return self.out_proj(x)


def main() -> None:
    from train_parser import TrainParser

    parser = TrainParser()

    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=64)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    def cosine_lr(step: int, base_lr: float) -> float:
        min_lr = base_lr * 0.05
        if step < args.warmup_steps:
            return min_lr + (step / args.warmup_steps) * base_lr
        else:
            return min_lr + 0.5 * (base_lr - min_lr) * (
                1 + math.cos(step * math.pi / args.max_steps)
            )

    model = Atlas(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        r=args.r,
        p=args.p,
        chunk_size=args.chunk_size,
    ).to(device)

    optimizers = [
        torch.optim.Muon(
            [param for param in model.parameters() if param.ndim == 2], lr=args.muon_lr
        ),
        torch.optim.AdamW(
            [param for param in model.parameters() if param.ndim != 2],
            lr=args.adam_lr,
            betas=(0.9, 0.95),
        ),
    ]

    dl = DataLoader(
        datapath=args.data_dir, B=args.batch_size, T=args.seq_len, device=device
    )

    val_dl = (
        DataLoader(
            datapath=args.val_data_dir, B=args.batch_size, T=args.seq_len, device=device
        )
        if args.val_data_dir is not None
        else None
    )

    train.train(
        model=model,
        optimizers=optimizers,
        dl=dl,
        val_dl=val_dl,
        device=device,
        compile=args.compile,
        compile_mode=args.compile_mode,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr_schedule=cosine_lr,
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        val_every=args.val_every,
        val_batches=args.val_batches,
        save_every=args.save_every,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_project_name="Atlas",
        wandb_watch=args.wandb_watch,
        wandb_config=args,
    )


if __name__ == "__main__":
    main()
