import os
import numpy as np
import torch


class DataLoader:
    def __init__(
        self,
        datapath: str,
        B: int,
        T: int,
        current_shard: int | None = None,
        current_pos: int | None = None,
        device: torch.device | None = None,
    ):
        self.datapath = datapath
        self.B = B
        self.T = T
        self.device = device
        
        self.shard_files = sorted(os.listdir(datapath))
        self.num_shards = len(self.shard_files)
        if current_shard and current_pos:
            self.current_shard = current_shard
            self.current_pos = current_pos
            self.data = self.read_numpy(self.shard_files[self.current_shard])
        else:
            self.shard_reset()

    def next_shard(self):
        self.current_shard += 1
        if self.current_shard >= self.num_shards:
            self.shard_reset()
        self.current_pos = 0
        self.data = self.read_numpy(self.shard_files[self.current_shard])
        print(f"Dataloader now on shard {self.current_shard} of {self.num_shards - 1}")

    def shard_reset(self) -> None:
        self.current_shard = 0
        self.current_pos = 0
        self.data = self.read_numpy(self.shard_files[self.current_shard])
        print("Dataloader reset")

    def read_numpy(self, path: str) -> torch.Tensor:
        data = np.load(self.datapath + path)
        return torch.from_numpy(data).long().to(self.device)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_pos + self.B * self.T + 1 >= len(self.data):
            self.next_shard()
        buf = self.data[self.current_pos: self.current_pos + self.B * self.T + 1]
        self.current_pos += self.B * self.T
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)