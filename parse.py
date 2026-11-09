import argparse
from p2pfl.utils.topologies import TopologyType

def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL DFL experiment argument")
    
    parser.add_argument("--nodes", type=int, help="The number of nodes", default=4)
    parser.add_argument("--rounds", type=int, help="The number of rounds", default=2)
    parser.add_argument("--epochs", type=int, help="The number of epochs", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics", default=False)
    parser.add_argument("--seed", type=int, help="The seed to use.", default=666)
    parser.add_argument("--batch_size", type=int, help="The batch size for training.", default=128)
    parser.add_argument("--reduced_dataset", action="store_true", help="Use a reduced dataset just for testing.", default=True)
    parser.add_argument("--topology", type=str, choices=[t.value for t in TopologyType], help="Topology to use", default="full")
    
    args = parser.parse_args()
    args.topology = TopologyType(args.topology)
    
    return args