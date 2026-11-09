import matplotlib.pyplot as plt

from p2pfl.node import Node
from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.management.logger import logger
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.utils.utils import wait_convergence, wait_to_finish

def create_node(
    n: int, # The number of nodes
    r: int, # The number of rounds
    e: int, # The number of epochs
    batch_size: int = 128,
    show_metrics: bool = True,
    reduced_dataset: bool = True,
    topology: TopologyType = TopologyType.FULL
) -> None:
    
    # Dataset preparation here
    data = P2PFLDataset.from_huggingface("p2pfl/mnist")
    data.set_batch_size(batch_size)
    
    partition = data.generate_partitions(
        n * 50 if reduced_dataset else n,
        RandomIIDPartitionStrategy
    )
    
    nodes = []
    for i in range(n):
        address = "127.0.0.1"
        
        node = Node(
            model_build_fn(),
            partition[i],
            addr=address,
            protocol=GrpcCommunicationProtocol(),
            aggregator=FedAvg(),
        )
        node.start()
        nodes.append(node)
        
    malicious_node = nodes[n-1] # We identify the malicious node
    logger.info(malicious_node.addr, "I am the Malicious Node👺")
    try:
        
        # Create a topology matrix and connect the nodes with that topology
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        
        # We wait until all nodes are connected
        wait_convergence(
            nodes=nodes, 
            n_neis=n-1,
            wait=60, # in seconds
            only_direct=True,
            debug=True
        )
        
        if r < 1:
            raise ValueError("The Amount of round is less than 1. No training")
        
        # Start Decentralized Federated Learning
        nodes[0].set_start_learning(rounds=r, epochs=e)
        
        # Wait until all nodes have finished the workflow
        wait_to_finish(
            nodes=nodes, 
            timeout=60 * 60, # 1hr
            debug=True
        )
        
        # Local Logs ---------- From the github docs
        if show_metrics:
            local_logs = logger.get_local_logs()
            if local_logs != {}:
                logs_l = list(local_logs.items())[0][1]
                #  Plot experiment metrics
                for round_num, round_metrics in logs_l.items():
                    for node_name, node_metrics in round_metrics.items():
                        for metric, values in node_metrics.items():
                            x, y = zip(*values, strict=False)
                            plt.plot(x, y, label=metric)
                            # Add a red point to the last data point
                            plt.scatter(x[-1], y[-1], color="red")
                            plt.title(f"Round {round_num} - {node_name}")
                            plt.xlabel("Epoch")
                            plt.ylabel(metric)
                            plt.legend()
                            plt.show()

            # Global Logs
            global_logs = logger.get_global_logs()
            if global_logs != {}:
                logs_g = list(global_logs.items())[0][1]  # Accessing the nested dictionary directly
                # Plot experiment metrics
                for node_name, node_metrics in logs_g.items():
                    for metric, values in node_metrics.items():
                        x, y = zip(*values, strict=False)
                        plt.plot(x, y, label=metric)
                        # Add a red point to the last data point
                        plt.scatter(x[-1], y[-1], color="red")
                        plt.title(f"{node_name} - {metric}")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.show()
        
        
    except Exception as e:
        raise e
    finally:
        # We stop all nodes
        for node in nodes:
            node.stop()
            
            
            
        
        