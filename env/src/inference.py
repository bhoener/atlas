import torch
import torch.nn.functional as F
from atlas import Atlas
import tiktoken as tk


def main() -> None:
    class colors:
        RED = "\033[31m"
        GREEN = "\033[32m"
        BLUE = "\033[34m"
        RESET = "\033[0m"

    print("Atlas Model inference")
    if torch.cuda.is_available:
        device = torch.device("cuda")
        print(f"Using {colors.GREEN}CUDA{colors.RESET}")
    else:
        device = torch.device("cpu")
        print("Using cpu")

    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")

    temperature = 0.7
    top_k = 50

    d_model = 768
    n_heads = 16
    n_layers = 12
    vocab_size = 50304
    seq_len = 512
    r = 8
    p = 2
    chunk_size = 64

    save_path = "models/model_000003k.pth"

    model = Atlas(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        r=r,
        p=p,
        chunk_size=chunk_size,
    )

    d = model.poly_dim

    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in torch.load(save_path).items()
    }

    model.load_state_dict(state_dict)
    model = model.to(device)

    enc = tk.get_encoding("gpt2")

    states = [
        (
            torch.randn(d, d * 4, device=device),
            torch.randn(d * 4, d_model // n_heads, device=device),
            torch.zeros(d, d * 4, device=device),
            torch.zeros(d * 4, d_model // n_heads, device=device),
        )
        for _ in range(n_layers)
    ]

    while (
        prompt := input(colors.BLUE + "\n\nEnter a prompt: " + colors.RESET).strip()
    ) not in {"q", "quit"}:
        print(prompt, end="")

        tokens = [50256] + enc.encode(prompt)

        for _ in range(seq_len - len(tokens)):
            tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            states, logits = model(tokens_tensor, states)

            logits = logits[0, -1]

            mask = torch.ones_like(logits, device=device, dtype=bool)
            mask[torch.topk(logits, top_k).indices] = False
            logits[mask] = -float("inf")

            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            selected_idx = torch.multinomial(probs, num_samples=1).item()
            print(enc.decode([selected_idx]), end="")

            tokens += [selected_idx]

    print()
    print(colors.RED + "Exiting" + colors.RESET)


if __name__ == "__main__":
    main()
