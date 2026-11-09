import time
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.examples.mnist.model.mlp_pytorch import MLP
from p2pfl.node import Node

# Node 2 creation
node2 = Node(
    model=LightningModel(MLP()),
    data=P2PFLDataset.from_huggingface("p2pfl/MNIST"),
    addr=f"127.0.0.1:{7000}",
)
node2.start()

## Connect node2 with node1 for communication
node2.connect(f"127.0.0.1:{6000}")

time.sleep(4)

## We then start the distributed node learning
node2.set_start_learning(rounds=3, epochs=2)

while True:
    time.sleep(1)
    if node2.state.round is None:
        break
node2.stop()

