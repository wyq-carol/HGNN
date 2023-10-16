import torch
import torch.nn as nn
import warnings
import copy
import os
import json
import sys
import shutil
from tqdm import tqdm
import dgl

sys.path.append('/raid/wangyiqing/workspace/HGNN/HGNN/')

from utils.utils import set_random_seed, convert_to_gpu
from utils.EarlyStopping import EarlyStopping
from utils.utils import count_parameters_in_KB, load_dataset, get_node_data_loader, get_optimizer_and_lr_scheduler
from RGCN import RGCN
from utils.Classifier import Classifier
from utils.utils import evaluate_node_classification


args = {
    'dataset': 'OGB_MAG',  # OGB_MAG, OAG_CS_Field_F1, OAG_CS_Field_F2, OAG_CS_Venue, Amazon
    'model_name': 'RGCN_node_classification_lr0.001_dropout0.3',
    'seed': 0,
    'cuda': 0,
    'learning_rate': 0.001,
    'hidden_units': [256, 256],
    'dropout': 0.3,
    'n_bases': -1,  # number of filter weight matrices, default: -1 [use all]
    'use_self_loop': True,  # include self feature as a special relation, True in the original paper
    'batch_size': 2560,  # the number of graphs to sample in each batch
    'node_neighbors_min_num': 10,  # number of sampled edges for each type for each GNN layer
    'optimizer': 'adam',
    'weight_decay': 0,
    'epochs': 2,
    'patience': 50
}
args['data_path'] = f'../../dataset/{args["dataset"]}/{args["dataset"]}.pkl'
args['data_split_idx_path'] = f'../../dataset/{args["dataset"]}/{args["dataset"]}_split_idx.pkl'
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    set_random_seed(args['seed'])

    if args['dataset'] == 'OGB_MAG' or args['dataset'] == 'OAG_CS_Field_F1' or args['dataset'] == 'OAG_CS_Field_F2' or args['dataset'] == 'OAG_CS_Venue':
        args['predict_category'] = 'paper'
    elif args['dataset'] == 'Amazon':
        args['predict_category'] = 'review'
    else:
        raise ValueError(f"wrong dataset {args['data_path']}")

    print(f'loading dataset {args["dataset"]}...')

    graph, labels, num_classes, train_idx, valid_idx, test_idx = load_dataset(data_path=args['data_path'],
                                                 predict_category=args['predict_category'],
                                                 data_split_idx_path=args['data_split_idx_path'])
    print(f"original graph: {graph}")
    print(f'get node data loader...')
    train_loader, _, _ = get_node_data_loader(args['node_neighbors_min_num'], len(args['hidden_units']),
                                                                 graph,
                                                                 batch_size=args['batch_size'],
                                                                 sampled_node_type=args['predict_category'],
                                                                 train_idx=train_idx, valid_idx=valid_idx,
                                                                 test_idx=test_idx)
    rgcn = RGCN(graph=graph, input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                hidden_sizes=args['hidden_units'], num_bases=args['n_bases'], dropout=args['dropout'], use_self_loop=args['use_self_loop'])

    classifier = Classifier(n_hid=args['hidden_units'][-1], n_out=num_classes)

    model = nn.Sequential(rgcn, classifier)

    model = convert_to_gpu(model, device=args['device'])
    print(model)

    print(f'the size of RGCN parameters is {count_parameters_in_KB(model[0])} KB.')

    print(f'configuration is {args}')

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model, args['optimizer'], args['learning_rate'],
                                                          args['weight_decay'],
                                                          steps_per_epoch=len(train_loader), epochs=args['epochs'])

    loss_func = nn.CrossEntropyLoss()

    train_steps = 0
    
    for epoch in range(args['epochs']):
        model.train()
        train_total_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, ncols=120)
        for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
            print(f"training for the {epoch}-th epoch {batch}-th batch")
            blocks = [convert_to_gpu(b, device=args['device']) for b in blocks]
            # nodes representation with all types in the heterogeneous graph
            input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}
            # Tensor, (samples_num, )
            train_y_true = convert_to_gpu(labels[output_nodes[args['predict_category']]],
                                          device=args['device'])
            # dictionary of all types of nodes representation
            
            nodes_representation = model[0](blocks, copy.deepcopy(input_features))
            train_y_predict = model[1](nodes_representation[args['predict_category']])

            loss = loss_func(train_y_predict, train_y_true)

            train_total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loader_tqdm.set_description(f'training for the {batch}-th batch, train loss: {loss.item()}')
            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)

        train_total_loss /= (batch + 1)

        print(f'Epoch: {epoch}, learning rate: {optimizer.param_groups[0]["lr"]}')
