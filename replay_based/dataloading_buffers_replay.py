from typing import Any
from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.utils.data_loader import GroupBalancedDataLoader

from avalanche.training.storage_policy import ReservoirSamplingBuffer
from types import SimpleNamespace

from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy

from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import SupervisedPlugin

benchmark = SplitMNIST(n_experiences=5, return_task_id=True)

dl = GroupBalancedDataLoader([exp.dataset for exp in benchmark.train_stream], batch_size=5)
for x,y,t in dl:
    print(x.shape, y.shape, t)
    print(t.tolist())
    break


benchmark = SplitMNIST(5, return_task_id=False)
storage_p = ReservoirSamplingBuffer(max_size=30)

print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")

for i in range(5):
    strategy_state = SimpleNamespace(experience=benchmark.train_stream[i])
    storage_p.update(strategy_state)
    print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
    print(f"class targets: {storage_p.buffer.targets.uniques}\n")

storage_p = ParametricBuffer(
    max_size=30,
    groupby='class',
    selection_strategy=RandomExemplarsSelectionStrategy()
)

print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
for i in range(5):
    strategy_state = SimpleNamespace(experience=benchmark.train_stream[i])
    storage_p.update(strategy_state)
    print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
    print(f"class targets: {storage_p.buffer.targets.uniques}\n")

for k, v in storage_p.buffer_groups.items():
    print(f"(group {k}) -> size {len(v.buffer)}")

datas = [v.buffer for v in storage_p.buffer_groups.values()]
dl = GroupBalancedDataLoader(datas)

for x, y, t in dl:
    print(y.tolist())
    break

class CustomReplayPlugin(SupervisedPlugin):
    def __init__(self, storage_policy):
        super().__init__()
        self.storage_policy = storage_policy

    def before_training_exp(self, strategy, num_workers: int=0, shuffle:bool=True, **kwargs) -> Any:
        '''Here we set the dataloader.'''
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't need to use the buffer, no need to change the dataloader.
            return
        
        # Replay dataloader samples mini-batches from the memory and current
        # data separately and combines them together.
        print("Override the dataloader.")
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle
        )

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs) -> Any:
        '''We update the buffer after each experience.
           You can use a different callback to update the buffer in a different place.'''
        print("Buffer update.")
        self.storage_policy.update(strategy, **kwargs)

from torch.nn import CrossEntropyLoss
from avalanche.training import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
import torch

scenario = SplitMNIST(5)
model = SimpleMLP(num_classes=scenario.n_classes)
stroage_p = ParametricBuffer(
    max_size=500,
    groupby='class',
    selection_strategy=RandomExemplarsSelectionStrategy()
)

# choose some metric and evaluation method
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    loggers=[interactive_logger]
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, torch.optim.Adam(model.parameters(), lr=0.001),
    CrossEntropyLoss(), train_mb_size=100, train_epochs=1,
    eval_mb_size=100, plugins=[CustomReplayPlugin(storage_p)],
    evaluator=eval_plugin, device="cuda:0"
)

# TRAINING LOOP
print("Starting experiment...")
results = []
for experience in scenario.train_stream:
    print("Start of experience", experience.current_experience)
    cl_strategy.train(experience)
    print("Training completed")

    print("Computing accuracy on the whole test set")
    results.append(cl_strategy.eval(scenario.test_stream))

print('the last result', results[-1])