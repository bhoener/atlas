import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import DataLoader
import math
import train

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        assert d_model % n_heads == 0
        
        self.d_key = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.wo = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        Q = self.wq(x).view(B, L, self.n_heads, self.d_key).transpose(1, 2)
        K = self.wk(x).view(B, L, self.n_heads, self.d_key).transpose(1, 2)
        V = self.wv(x).view(B, L, self.n_heads, self.d_key).transpose(1, 2)
        
        attn_scores = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        attn_scores = attn_scores.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        
        return self.wo(attn_scores)
    
class MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out
        
        self.W = nn.Linear(d_in, d_h)
        self.V = nn.Linear(d_in, d_h)
        
        self.act = nn.Sigmoid()
        self.l2 = nn.Linear(d_h, d_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.act(self.W(x)) * self.V(x))
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_model * 4, d_model)
        
        self.rms1 = nn.RMSNorm(d_model)
        self.rms2 = nn.RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.rms1(x))
        x = x + self.mlp(self.rms2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        self.layers = nn.ModuleList(DecoderBlock(d_model=d_model, n_heads=n_heads) for _ in range(self.n_layers))
        
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.size()
        x = self.emb(x)
        
        x = x + self.pos_emb(torch.arange(L, device=x.device))
        
        for layer in self.layers:
            x = layer(x)
            
        return self.out_proj(x)

def main() -> None:
    from train_parser import TrainParser

    parser = TrainParser()
    
    parser.add_argument("--is_transformer", type=bool, default=True)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    
    def cosine_lr(step: int, base_lr: float) -> float:
        min_lr = base_lr * 0.05
        if step < args.warmup_steps:
            return min_lr + (step / args.warmup_steps) * base_lr
        else:
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(step * math.pi / args.max_steps))

    model = Transformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
    ).to(device)

    optimizers = [
        torch.optim.Muon(
            [param for param in model.parameters() if param.ndim == 2], lr=args.muon_lr
        ),
        torch.optim.AdamW(
            [param for param in model.parameters() if param.ndim != 2], lr=args.adam_lr, betas=(0.9, 0.95)
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