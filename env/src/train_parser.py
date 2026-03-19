import argparse

class TrainParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--d_model", type=int, default=768)
        self.add_argument("--n_heads", type=int, default=16)
        self.add_argument("--n_layers", type=int, default=12)
        self.add_argument("--vocab_size", type=int, default=50304)
        self.add_argument("--seq_len", type=int, default=512)
        
        self.add_argument("--device", type=str, default="cuda")

        self.add_argument("--max_steps", type=int, default=10000)
        self.add_argument("--batch_size", type=int, default=8)
        self.add_argument("--grad_accum_steps", type=int, default=2)

        self.add_argument("--adam_lr", type=float, default=3e-4)
        self.add_argument("--muon_lr", type=float, default=0.001)
        self.add_argument("--warmup_steps", type=int, default=200)

        self.add_argument("--log_every", type=int, default=10)
        self.add_argument("--val_every", type=int, default=1000)
        self.add_argument("--val_batches", type=int, default=4)
        self.add_argument("--save_every", type=int, default=1000)

        self.add_argument("--save_dir", type=str, default="models/")
        self.add_argument("--data_dir", type=str, default="fineweb/")
        self.add_argument("--val_data_dir", type=str, default=None)

        self.add_argument("--compile", type=bool, default=True)
        self.add_argument("--compile_mode", type=str, default="default")

        self.add_argument("--use_wandb", type=bool, default=True)
        self.add_argument("--wandb_watch", type=bool, default=False)
        
        self.add_argument("--wandb_project_name", type=str, default=None)