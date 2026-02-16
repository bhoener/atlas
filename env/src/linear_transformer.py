import torch
import torch.nn as nn
import torch.nn.functional as F
import train
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
        ).to(A.device),
        torch.randn(
            A.size()[:-2]
            + (
                A.size(-1),
                r,
            ),
            generator=gen,
        ).to(A.device),
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
    D_tilde = 1.0 + qk_similarity @ torch.ones(L, 1).to(Q.device)
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

    D_tilde = 1.0 + fast_lt_mul(phi_Q, phi_K, torch.ones(B, H, L, 1).to(Q.device))
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
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--r", type=int, default=64)
    parser.add_argument("--p", type=int, default=2)
    
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)

    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--muon_lr", type=float, default=3e-6)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1000)

    parser.add_argument("--save_dir", type=str, default="models/")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--val_data_dir", type=str, default=None)

    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_watch", type=bool, default=True)
    
    parser.add_argument("--wandb_project_name", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = PolySketchFormer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        r=args.r,
        p=args.p,
    ).to(device)

    optimizers = [
        torch.optim.Muon(
            [param for param in model.parameters() if param.ndim == 2], lr=args.muon_lr
        ),
        torch.optim.AdamW(
            [param for param in model.parameters() if param.ndim != 2], lr=args.adam_lr
        ),
    ]

    dl = DataLoader(datapath=args.data_dir, B=args.batch_size, T=args.seq_len, device=device)

    val_dl = (
        DataLoader(datapath=args.val_data_dir, B=args.batch_size, T=args.seq_len, device=device)
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
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        val_every=args.val_every,
        val_batches=args.val_batches,
        save_every=args.save_every,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_project_name="PolySketchFormer",
        wandb_watch=args.wandb_watch,
        wandb_config=args,
    )


if __name__ == "__main__":
    main()
