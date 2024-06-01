from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader

from avalanche.training.storage_policy import ReservoirSamplingBuffer
from types import SimpleNamespace

benchmark = SplitMNIST(n_experiences=5, return_task_id=True)

dl = GroupBalancedDataLoader([exp.dataset for exp in benchmark.train_stream], batch_size=5)
for x,y,t in dl:
    print(x.shape, y.shape, t)
    print(t.tolist())
    break


benchmark = SplitMNIST(5, return_task_id=False)
storage_p = ReservoirSamplingBuffer(max_size=30)

print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
