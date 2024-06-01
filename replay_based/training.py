from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC # and more

from avalanche.benchmarks.classic import SplitMNIST

from avalanche.training.plugins import EarlyStoppingPlugin

from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import ReplayPlugin, EWCPlugin

from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer


model = SimpleMLP(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()
cl_strategy = Naive(
    model, optimizer, criterion,
    train_mb_size = 100, train_epochs = 4, eval_mb_size = 100
)

# scenario
benchmark = SplitMNIST(n_experiences=5, seed=1)

# training loop
print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print('start of experience:', experience.current_experience)
    print('Current Classes:', experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(benchmark.test_stream))

print('66666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666')

strategy = Naive(
    model, optimizer, criterion,
    plugins = [EarlyStoppingPlugin(patience=3, val_stream_name='train')]
)

replay = ReplayPlugin(mem_size=100)
ewc = EWCPlugin(ewc_lambda=0.001)
strategy = SupervisedTemplate(
    model, optimizer, criterion,
    plugins = [replay, ewc]
)

class ReplayP(SupervisedPlugin):

    def __init__(self, mem_size):
        ''' A simple replay plugin with reservoir sampling.'''
        super().__init__()
        self.buffer = ReservoirSamplingBuffer(max_size=mem_size)

    def before_training_exp(self, strategy: "SupervisedTemplate", num_workers:int=0, shuffle:bool=True, **kwargs):
        ''' Use a custom dataloader to combine samples from the current data and memory buffer. '''
        if len(self.buffer.buffer) == 0:
            # first experience. We don't use the buffer, no need to change the dataloader.
            return
        strategy.dataloader = ReplayDataLoader(strategy.adapted_dataset, self.buffer.buffer, oversample_small_tasks=True, num_workers=num_workers, batch_size=strategy.train_mb_size, shuffle=shuffle)

    def after_training_exp(self, strategy:"SupervisedTemplate", **kwargs):
        ''' Update the buffer. '''
        self.buffer.update(strategy, **kwargs)

benchmark = SplitMNIST(n_experiences=5, seed=1)
model = SimpleMLP(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()
strategy = Naive(model=model, optimizer=optimizer, criterion=criterion, train_mb_size=128, plugins=[ReplayP(mem_size=2000)])
strategy.train(benchmark.train_stream)
strategy.eval(benchmark.test_stream)
        
        