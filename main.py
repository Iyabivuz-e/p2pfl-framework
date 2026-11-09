import parse
from nodes.node import create_node
from p2pfl.settings import Settings
from p2pfl.utils.utils import set_standalone_settings

if __name__=="__main__":
    # parse the args in the cmd
    args = parse.__parse_args()

    
    ## Set the settings -- heartbeats, grpc, gossip, etc...
    set_standalone_settings()
    
    if args.seed is None:
        Settings.general.SEED = args.seed
        
    create_node(
        args.nodes, 
        args.rounds, 
        args.epochs, 
        )