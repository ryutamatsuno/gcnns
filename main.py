from data.data import load_data
from models.gcn import create_gcn_model
from models.gat import create_gat_model
from models.sgc import create_sgc_model
from models.gfnn import create_gfnn_model
#from models.graphsage import create_graphsage_model
from models.masked_gcn import create_masked_gcn_model
from train import run
from utils import preprocess_features


import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='small')
    parser.add_argument('--model', type=str, default='sgc')
    parser.add_argument('--niter', type=int, default=10)

    args = parser.parse_args()


    data = load_data(args.dataset)
    data.features = preprocess_features(data.features)

    if args.model == 'sgc':
        model, optimizer = create_sgc_model(data)
    elif args.model == 'gcn':
        model, optimizer = create_gcn_model(data)
    else:
        raise ValueError(args.model)
    run(data, model, optimizer, verbose=False, niter=args.niter, patience=10)
