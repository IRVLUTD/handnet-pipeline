import itertools
from torch.utils.data import Subset


class ConcatDataloader:
    def __init__(self, dataloaders):
        self.loaders = dataloaders

    def __iter__(self):
        self.iters = [iter(loader) for loader in self.loaders]
        self.idx_cycle = itertools.cycle(list(range(len(self.loaders))))
        return self

    def __next__(self):
        loader_idx = next(self.idx_cycle)
        loader = self.iters[loader_idx]
        batch = next(loader)
        if isinstance(loader._dataset, Subset):
            dataset = loader._dataset.dataset
        else:
            dataset = loader._dataset
        dat_name = dataset.pose_dataset.name
        batch["dataset"] = dat_name
        batch["root"] = "wrist"
        batch["split"] = dataset.pose_dataset.split

        return batch

    def __len__(self):
        return min(len(loader) for loader in self.loaders) * len(self.loaders)
