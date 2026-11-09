from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.examples.mnist.model.mlp_pytorch import MLP
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.node import Node


# Node 1 creation
node1 = Node(
    model=LightningModel(MLP()),
    data=P2PFLDataset.from_huggingface("p2pfl/MNIST"),
    addr=f"127.0.0.1:{6000}",
    # Others later
)
node1.start()

input("Press any key to stop\n")
node1.stop()