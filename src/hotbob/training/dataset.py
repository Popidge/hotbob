from __future__ import annotations

from torch.utils.data import Dataset

from hotbob.types import TaskTrace


class TraceDataset(Dataset[TaskTrace]):
    def __init__(self, traces: list[TaskTrace]) -> None:
        self.traces = traces

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> TaskTrace:
        return self.traces[idx]
