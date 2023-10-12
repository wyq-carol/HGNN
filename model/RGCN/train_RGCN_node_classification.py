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
    'epochs': 1,
    'patience': 50
}
args['data_path'] = f'../../dataset/{args["dataset"]}/{args["dataset"]}.pkl'
args['data_split_idx_path'] = f'../../dataset/{args["dataset"]}/{args["dataset"]}_split_idx.pkl'
# breakpoint()
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'


def evaluate(model: nn.Module, loader: dgl.dataloading.DataLoader, loss_func: nn.Module,
             labels: torch.Tensor, predict_category: str, device: str, mode: str):
    """

    :param model: model
    :param loader: data loader (validate or test)
    :param loss_func: loss function
    :param labels: node labels
    :param predict_category: str
    :param device: device str
    :param mode: str, evaluation mode, validate or test
    :return:
    """
    model.eval()
    with torch.no_grad():
        y_trues = []
        y_predicts = []
        total_loss = 0.0
        loader_tqdm = tqdm(loader, ncols=120)
        for batch, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            # nodes representation with all types in the heterogeneous graph
            input_features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in blocks[0].ntypes}
            # Tensor, (samples_num, )
            y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)
            nodes_representation = model[0](blocks, copy.deepcopy(input_features))
            y_predict = model[1](nodes_representation[predict_category])

            loss = loss_func(y_predict, y_true)

            total_loss += loss.item()
            y_trues.append(y_true.detach().cpu())
            y_predicts.append(y_predict.detach().cpu())

            loader_tqdm.set_description(f'{mode} for the {batch}-th batch, {mode} loss: {loss.item()}')

        total_loss /= (batch + 1)
        y_trues = torch.cat(y_trues, dim=0)
        y_predicts = torch.cat(y_predicts, dim=0)

    return total_loss, y_trues, y_predicts


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
    
    print(f'get node data loader...')
    train_loader, val_loader, test_loader = get_node_data_loader(args['node_neighbors_min_num'], len(args['hidden_units']),
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

    save_model_folder = f"./save_model/{args['dataset']}/{args['model_name']}"

    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                   save_model_name=args['model_name'])

    loss_func = nn.CrossEntropyLoss()

    train_steps = 0
    
    for epoch in range(args['epochs']):
        model.train()

        train_y_trues = []
        train_y_predicts = []
        train_total_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, ncols=120)
        for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
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
            train_y_trues.append(train_y_true.detach().cpu())
            train_y_predicts.append(train_y_predict.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loader_tqdm.set_description(f'training for the {batch}-th batch, train loss: {loss.item()}')

            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)

        train_total_loss /= (batch + 1)
        train_y_trues = torch.cat(train_y_trues, dim=0)
        train_y_predicts = torch.cat(train_y_predicts, dim=0)
        
        train_accuracy, train_micro_f1, train_macro_f1 = evaluate_node_classification(predicts=train_y_predicts.argmax(dim=1), labels=train_y_trues)

        model.eval()

        val_total_loss, val_y_trues, val_y_predicts = evaluate(model, loader=val_loader, loss_func=loss_func,
                                                               labels=labels,
                                                               predict_category=args['predict_category'],
                                                               device=args['device'],
                                                               mode='validate')

        val_accuracy, val_micro_f1, val_macro_f1 = evaluate_node_classification(predicts=val_y_predicts.argmax(dim=1),
                                                                                labels=val_y_trues)

        test_total_loss, test_y_trues, test_y_predicts = evaluate(model, loader=test_loader, loss_func=loss_func,
                                                                  labels=labels,
                                                                  predict_category=args['predict_category'],
                                                                  device=args['device'],
                                                                  mode='test')

        test_accuracy, test_micro_f1, test_macro_f1 = evaluate_node_classification(
            predicts=test_y_predicts.argmax(dim=1),
            labels=test_y_trues)

        print(
            f'Epoch: {epoch}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}, '
            f'accuracy {train_accuracy:.4f}, micro f1 {train_micro_f1:.4f}, macro f1 {train_macro_f1:.4f}, \n'
            f'valid loss: {val_total_loss:.4f}, '
            f'accuracy {val_accuracy:.4f}, micro f1 {val_micro_f1:.4f}, macro f1 {val_macro_f1:.4f} \n'
            f'test loss: {test_total_loss:.4f}, '
            f'accuracy {test_accuracy:.4f}, micro f1 {test_micro_f1:.4f}, macro f1 {test_macro_f1:.4f}')

        early_stop = early_stopping.step([('accuracy', val_accuracy, True), ('macro_f1', val_macro_f1, True)], model)

        if early_stop:
            break

    # load best model
    early_stopping.load_checkpoint(model)

    print('performing model inference...')
    # evaluate the best model
    model.eval()
    nodes_representation = model[0].inference(graph, copy.deepcopy(
        {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}), device=args['device'])

    train_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[train_idx]
    train_y_trues = convert_to_gpu(labels[train_idx], device=args['device'])
    train_accuracy, train_micro_f1, train_macro_f1 = evaluate_node_classification(predicts=train_y_predicts.argmax(dim=1),
                                                                                    labels=train_y_trues)

    print(f'final train accuracy: {train_accuracy:.4f}, micro f1 {train_micro_f1:.4f}, macro f1 {train_macro_f1:.4f}')

    val_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        valid_idx]
    val_y_trues = convert_to_gpu(labels[valid_idx], device=args['device'])
    val_accuracy, val_micro_f1, val_macro_f1 = evaluate_node_classification(predicts=val_y_predicts.argmax(dim=1),
                                                                            labels=val_y_trues)

    print(f'final valid accuracy {val_accuracy:.4f}, micro f1 {val_micro_f1:.4f}, macro f1 {val_macro_f1:.4f}')

    test_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        test_idx]
    test_y_trues = convert_to_gpu(labels[test_idx], device=args['device'])
    test_accuracy, test_micro_f1, test_macro_f1 = evaluate_node_classification(predicts=test_y_predicts.argmax(dim=1),
                                                                               labels=test_y_trues)
    print(f'final test accuracy {test_accuracy:.4f}, micro f1 {test_micro_f1:.4f}, macro f1 {test_macro_f1:.4f}')

    # save model result
    result_json = {
        "train accuracy": float(f"{train_accuracy:.4f}"), "train micro f1": float(f"{train_micro_f1:.4f}"), "train macro f1": float(f"{train_macro_f1:.4f}"),
        "validate accuracy": float(f"{val_accuracy:.4f}"), "validate micro f1": float(f"{val_micro_f1:.4f}"), "validate macro f1": float(f"{val_macro_f1:.4f}"),
        "test accuracy": float(f"{test_accuracy:.4f}"), "test micro f1": float(f"{test_micro_f1:.4f}"), "test macro f1": float(f"{test_macro_f1:.4f}")
    }
    result_json = json.dumps(result_json, indent=4)

    save_result_folder = f"./results/{args['dataset']}"
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args['model_name']}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)

    # sys.exit()
