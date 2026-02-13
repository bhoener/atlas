import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from dataloader import DataLoader


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.size())


def poly_sketch_with_negativity(
    A: torch.Tensor,
    r: int,
    p: int,
    deterministic: bool = False,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    if p == 1:
        return A
    gen = torch.Generator().manual_seed(42) if gen is None and deterministic else gen
    M_1 = poly_sketch_with_negativity(A, r, p // 2, gen)
    M_2 = poly_sketch_with_negativity(A, r, p // 2, gen)

    G_1, G_2 = (
        torch.randn(
            A.size()[:-2]
            + (
                A.size(-1),
                r,
            ),
            generator=gen,
        ),
        torch.randn(
            A.size()[:-2]
            + (
                A.size(-1),
                r,
            ),
            generator=gen,
        ),
    )
    return (1 / r) ** 0.5 * ((M_1 @ G_1) * (M_2 @ G_2))


def poly_sketch_non_negative(
    A: torch.Tensor, r: int, p: int, deterministic: bool = False
) -> torch.Tensor:
    M = poly_sketch_with_negativity(A, r, p // 2, deterministic=deterministic)
    # paper says M⊗2 ∈ R^{k×r^2}
    # and M ∈ R^{k×r}
    # -> (k, r), (k, r) -> (k, r, r) -> (k, r^2)
    M = torch.einsum("...i, ...j -> ...ij", M, M).squeeze(-1)
    res = torch.einsum("...i, ...j -> ...", M, M)
    return res


def naive_causal_polysketch_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    h: int = 4,
    r: int = 8,
    p: int = 4,
    deterministic: bool = False,
) -> torch.Tensor:
    B, L, D = Q.size()
    H = D // h
    Q, K, V = (
        F.layer_norm(Q.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
        F.layer_norm(K.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
        F.layer_norm(V.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
    )
    phi_Q, phi_K = (
        poly_sketch_non_negative(Q, r, p, deterministic=deterministic),
        poly_sketch_non_negative(K, r, p, deterministic=deterministic),
    )  # (B, L, H, h) -> (B, L, H, r^2)
    qk_similarity = torch.tril(
        phi_Q @ phi_K.permute(0, 1, 3, 2)
    )  # (B, H, L, r^2), (B, H, r^2, L) -> (B, H, L, L)
    D_tilde = 1.0 + qk_similarity @ torch.ones(L, 1)
    # (B, H, L, L), (L, 1) -> (B, H, L, 1)

    return D_tilde**-1.0 * (
        qk_similarity @ V
    )  # (B, H, L, L), (B, H, L, h) -> (B, H, L, h)


def causal_polysketch_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    h: int = 4,
    r: int = 8,
    p: int = 4,
    deterministic: bool = False,
) -> torch.Tensor:
    B, L, D = Q.size()
    H = D // h
    Q, K, V = (
        norm(Q.view(B, L, H, h).transpose(1, 2)),
        norm(K.view(B, L, H, h).transpose(1, 2)),
        norm(V.view(B, L, H, h).transpose(1, 2)),
    )  # (B, H, L, h)

    phi_Q, phi_K = (
        poly_sketch_non_negative(Q, r, p, deterministic=deterministic),
        poly_sketch_non_negative(K, r, p, deterministic=deterministic),
    )  # (B, H, L, r^2)

    D_tilde = 1.0 + fast_lt_mul(phi_Q, phi_K, torch.ones(B, H, L, 1))
    # (B, H, L, r^2) @ (B, H, r^2 , L) -> (B, H, L, L), (B, H, L, 1) -> (B, H, L, 1)

    return D_tilde**-1.0 * fast_lt_mul(phi_Q, phi_K, V)  # (B, H, L, h)


def fast_lt_mul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: int = 16,
) -> torch.Tensor:
    batch, H, L, r_sq = A.size()
    h = C.size(-1)

    t = L // b
    out = torch.empty(batch, H, L, h)

    Z_l = torch.zeros(batch, H, r_sq, h)
    for l in range(t):
        idx = torch.arange(l * b, (l + 1) * b)

        A_l, B_l, C_l = (
            A[:, :, idx],  # (B, H, b, r^2)
            B[:, :, idx],  # (B, H, b, r^2)
            C[:, :, idx],  # (B, H, b, h) or (B, H, b, 1)
        )

        H_l = torch.einsum(
            "BHbr, BHbh -> BHrh", B_l, C_l
        )  # (B, b, H, r^2), (B, b, H, h) or (B, b, 1, 1) -> (B, H, r^2, h)

        P_l = torch.tril(A_l @ B_l.permute(0, 1, 3, 2)) @ C_l

        # (B, H, b, r^2), (B, H, r^2, b) -> (B, H, b, b)
        # (B, H, b, b), (B, H, b, h) -> (B, H, b, h)

        out[:, :, idx, :] = P_l + A_l @ Z_l
        # (B, H, b, r^2), (B, H, r^2, h) -> (B, H, b, h)

        Z_l = Z_l + H_l
    return out


class PolySketchAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, r: int = 16, p: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.d_key = d_model // n_heads

        self.r = r
        self.p = p

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)

        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        return self.wo(
            causal_polysketch_attention(Q, K, V, self.d_key, self.r, self.p)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, D),
        )


class MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out

        self.l1 = nn.Linear(d_in, d_h)
        self.act = nn.GELU()
        self.l2 = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.act(self.l1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, r: int = 16, p: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.r = r
        self.p = p

        self.attn = PolySketchAttention(d_model, n_heads, r, p)

        self.mlp = MLP(d_model, d_model * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class PolySketchFormer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        vocab_size: int = 128,
        seq_len: int = 1024,
        r: int = 16,
        p: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.emb = nn.Embedding(vocab_size, d_model, max_norm=1)
        self.pos_emb = nn.Embedding(seq_len, d_model, max_norm=1)

        self.decoder_stack = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, r, p) for _ in range(n_layers)]
        )

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.orthogonal_(self.out_proj.weight, (1 / self.d_model**0.5 * 0.1))

    def forward(self, x: torch.Tensor):
        B, L = x.size()
        x = self.emb(x)
        x = x + self.pos_emb(torch.arange(0, L).to(x.device))

        for layer in self.decoder_stack:
            x = layer(x)

        return self.out_proj(x)


def main() -> None:
    use_wandb = True

    if use_wandb:
        import wandb
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")
    print("Device and precision set")

    d_model = 256
    n_heads = 8
    n_layers = 8
    vocab_size = 50304
    seq_len = 128
    r = 64
    p = 2

    max_steps = 1000000
    batch_size = 4
    grad_accum_steps = 8
    adam_lr = 3e-4
    muon_lr = 3e-6

    log_every = 10
    save_every = 1000

    save_dir = "models/"

    data_dir = "data/"
    
    compile = True

    model = PolySketchFormer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        vocab_size=vocab_size,
        seq_len=seq_len,
        r=r,
        p=p,
    )
    
    if compile:
        model = torch.compile(model)

    print("Model created")

    config = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "r": r,
        "p": p,
        "max_steps": max_steps,
        "batch_size": batch_size * grad_accum_steps,
        "adam_lr": adam_lr,
        "muon_lr": muon_lr,
        "compile": compile,
    }

    if use_wandb:
        run = wandb.init(
            project="PolySketchFormer",
            config=config,
        )
        wandb.watch(model)
        print("Wandb run started")

    optim = torch.optim.Muon(
        [param for param in model.parameters() if param.ndim == 2], lr=adam_lr
    )
    optim2 = torch.optim.AdamW(
        [param for param in model.parameters() if param.ndim != 2], lr=muon_lr
    )
    print("Optimizers Created")

    dl = DataLoader(datapath=data_dir, B=batch_size, T=seq_len, device=device)

    print("Beginning Training")
    print("=" * 100)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(config)
    print("=" * 100)

    for step in range(max_steps):
        t0 = time.time()
        loss_accum = 0.0
        optim.zero_grad()
        optim2.zero_grad()
        for micro_step in range(grad_accum_steps):
            xs, ys = dl.next()

            logits = model(xs)

            loss = (
                F.cross_entropy(logits.view(-1, vocab_size), ys.view(-1))
                / grad_accum_steps
            )

            loss.backward()

            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            loss_accum += loss.detach().item()

        optim.step()
        optim2.step()

        time_delta = time.time() - t0
        if use_wandb:
            run.log(
                {"step": step, "loss": loss_accum, "time": time_delta, "norm": norm}
            )

        if step % log_every == 0:
            print(
                f"step {step} | loss {loss_accum:.4f} | norm {norm:.4f} | time {time_delta * 1000:.4f}ms"
            )
        if step % save_every == 0:
            torch.save(model.state_dict(), save_dir + f"model_{step:08d}.pth")


if __name__ == "__main__":
    main()
