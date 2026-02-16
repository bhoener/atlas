import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import DataLoader
import time


def train(
    model: nn.Module,
    optimizers: list[torch.optim.Optimizer],
    dl: DataLoader,
    val_dl: DataLoader | None = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    compile: bool = False,
    max_steps: int = 10000,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    log_every: int = 10,
    val_every: int = 1000,
    val_batches: int = 4,
    save_every: int = 1000,
    save_dir: int = "models/",
    use_wandb: bool = False,
    wandb_project_name: str | None = None,
    wandb_watch: bool = False,
    wandb_config: dict | None = None,
) -> None:
    if use_wandb:
        import wandb

    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")
    print("Device and precision set")

    if compile:
        model = torch.compile(model)

    if use_wandb:
        run = wandb.init(
            project=wandb_project_name
            if wandb_project_name is not None
            else model.__class__.__name__,
            config=wandb_config,
        )
        if wandb_watch:
            wandb.watch(model)
        print("Wandb run started")

    print("Beginning Training")
    print("=" * 100)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(wandb_config)
    print("=" * 100)

    for step in range(max_steps):
        t0 = time.time()
        loss_accum = 0.0
        for optim in optimizers:
            optim.zero_grad()
        for micro_step in range(grad_accum_steps):
            xs, ys = dl.next()

            logits = model(xs)

            loss = (
                F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1))
                / grad_accum_steps
            )

            loss.backward()

            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            loss_accum += loss.detach().item()

        for optim in optimizers:
            optim.step()

        time_delta = time.time() - t0
        if use_wandb:
            run.log(
                {"step": step, "loss": loss_accum, "time": time_delta, "norm": norm}
            )

        if step % val_every == 0 and val_dl is not None:
            with torch.no_grad():
                loss_accum = 0.0
                for _ in range(val_batches):
                    xs, ys = val_dl.next()

                    logits = model(xs)

                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1))

                    loss_accum += loss.item() / val_batches
                print(
                    f"step {step} | val loss {loss_accum:.4f} | norm {norm:.4f} | time {time_delta * 1000:.4f}ms"
                )
                if use_wandb:
                    run.log(
                        {"val_loss": loss_accum}
                    )
        elif step % log_every == 0:
            print(
                f"step {step} | loss {loss_accum:.4f} | norm {norm:.4f} | time {time_delta * 1000:.4f}ms"
            )

        if step % save_every == 0:
            torch.save(model.state_dict(), save_dir + f"model_{int(step /1e3):06d}k.pth")


if __name__ == "__main__":
    train()
