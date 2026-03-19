import torch
from atlas import Atlas
import tiktoken as tk

def main() -> None:
    class colors():
        RED = '\033[31m'
        GREEN = '\033[32m'
        BLUE = '\033[34m'
        RESET = '\033[0m'
    print("Atlas Model inference")
    if torch.cuda.is_available:
        device = torch.device("cuda")
        print(f"Using {colors.GREEN}CUDA{colors.RESET}")
    else:
        device = torch.device("cpu")
        print("Using cpu")
    
    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")
    
    
    d_model = 768
    n_heads = 16
    n_layers = 8
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
    
    state_dict = {k.replace("_orig_mod.", "") : v for k, v in torch.load(save_path).items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    
    enc = tk.get_encoding("gpt2")
    
    while (prompt := input(colors.BLUE + "Enter a prompt: " + colors.RESET).strip()) != "q":
        print(prompt)
        tokens = torch.tensor(enc.encode(prompt), device=device).unsqueeze(0)
        logits = model(tokens, )
        print(enc.decode([torch.argmax(logits[:, -1], dim=-1).item()]))
        
        
    print()
    print(colors.RED + "Exiting" + colors.RESET)

if __name__ == "__main__":
    main()