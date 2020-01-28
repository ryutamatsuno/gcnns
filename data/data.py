import os
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch

#from utils import add_self_loops, normalize_adj

def is_sparse(x: torch.Tensor) -> bool:
    """
    :param x:
    :return: True if x is sparse tensor else False
    """
    try:
        x._indices()
    except RuntimeError:
        return False
    return True

class SparseTensor(torch.Tensor):
    """
    NeverUse
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError

def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def edglist2adj(edge_list) -> torch.sparse.FloatTensor:
    weight = torch.ones(edge_list.size(1))
    v = weight #deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj


def preprocess_features(features):
    rowsum = features.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1
    features = features / rowsum
    return features


class Data(object):
    def __init__(self, adj: SparseTensor, edge_list: torch.Tensor, features: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor):
        self.adj = adj
        self.edge_list = edge_list
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1
        #
        # print(labels.shape)
        # print(features.shape)
        # print(edge_list.shape)
        # print(train_mask.shape)
        # print(val_mask.shape)
        # print(test_mask.shape)
        # # gen small one
        #
        # top_dir = "./data_set_exp4/small"
        # if os.path.exists(top_dir) == False:
        #     os.mkdir(top_dir)
        # # get largest component
        # edge_list = edge_list.numpy()
        # es = []
        # for i in range(edge_list.shape[1]):
        #     es.append((edge_list[0,i],edge_list[1,i]))
        # G = nx.Graph()
        # G.add_edges_from(es)
        # print("N:",len(nx.nodes(G)))
        # print("M:",len(nx.edges(G)))
        # print("number of componetns:", nx.number_connected_components(G))
        #
        # def get_largest_component(G):
        #     if nx.number_connected_components(G) == 1:
        #         return G
        #     Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        #     G = G.subgraph(Gcc[0])
        #     return G
        # G = get_largest_component(G)
        #
        #
        # # sampling
        # sample_size = 1000
        # chosen_nodes = np.random.permutation(np.array(G.nodes))[:sample_size]
        # G = nx.subgraph(G, chosen_nodes)
        # G = get_largest_component(G)
        # print("sampled size:",len(nx.nodes(G)))
        #
        # # fix
        # sampled_nodes = np.array(sorted(nx.nodes(G)), dtype=np.int)
        #
        # labels = labels.numpy()[sampled_nodes]
        # features = features.numpy()[sampled_nodes]
        #
        # num_nodes = len(sampled_nodes)
        # idx = np.arange(num_nodes)
        # idx = np.random.permutation(idx)
        #
        # train_num = int(0.3 * num_nodes)
        # valid_num = int(0.3 * num_nodes)
        # tests_num = num_nodes - train_num - valid_num
        #
        # train_mask = np.zeros(num_nodes, dtype=np.int)
        # train_mask[idx[:train_num]] = 1
        # train_mask = train_mask.astype(bool)
        #
        # valid_mask = np.zeros(num_nodes, dtype=np.int)
        # valid_mask[idx[train_num:train_num + valid_num]] = 1
        # valid_mask = valid_mask.astype(bool)
        #
        # tests_mask = np.zeros(num_nodes, dtype=np.int)
        # tests_mask[idx[train_num + valid_num:]] = 1
        # tests_mask = tests_mask.astype(bool)
        #
        # # mapping
        # remap = {}
        # for i in range(num_nodes):
        #     remap[sampled_nodes[i]] = i
        # G = nx.relabel_nodes(G,mapping=remap)
        # edge_list = np.array(nx.edges(G), dtype=np.int)
        # .transpose(1,0)
        # directed = torch.stack((edge_list[1], edge_list[0]), dim=0)
        # edge_list = torch.cat((edge_list, directed), dim=1)

        #
        # # saving
        # np.savetxt(top_dir + '/labels.csv', labels, '%d', delimiter=",")
        # np.savetxt(top_dir + '/features.csv', features, '%f', delimiter=",")
        # np.savetxt(top_dir + '/edge_list.edg', edge_list, '%d',delimiter=",")
        # np.savetxt(top_dir + '/train_mask.csv', train_mask, '%d', delimiter=",")
        # np.savetxt(top_dir + '/valid_mask.csv', valid_mask, '%d', delimiter=",")
        # np.savetxt(top_dir + '/tests_mask.csv', tests_mask, '%d', delimiter=",")
        #
        # exit()

    def to(self, device):
        self.adj = self.adj.to(device)
        self.edge_list = self.edge_list.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)

    @property
    def A(self):
        return self.adj

def load(name) -> Data:
    top_dir = "./data/" + name
    labels = np.loadtxt(top_dir + '/labels.csv', dtype=np.int, delimiter=",")
    features = np.loadtxt(top_dir + '/features.csv', dtype=np.float, delimiter=",")
    edge_list = np.loadtxt(top_dir + '/edge_list.edg', dtype=np.int,delimiter=",")
    train_mask = np.loadtxt(top_dir + '/train_mask.csv', dtype=np.bool, delimiter=",")
    valid_mask = np.loadtxt(top_dir + '/valid_mask.csv', dtype=np.bool, delimiter=",")
    tests_mask = np.loadtxt(top_dir + '/tests_mask.csv', dtype=np.bool, delimiter=",")

    labels = torch.tensor(labels, dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float)
    edge_list = torch.tensor(edge_list, dtype=torch.long)

    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    tests_mask = torch.tensor(tests_mask, dtype=torch.bool)
    adj = edglist2adj(edge_list)

    data = Data(adj,edge_list,features,labels,train_mask,valid_mask,tests_mask)
    return data



def load_data(dataset_str: str, seed=None) -> Data:
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        data = load_planetoid_data(dataset_str)
    elif dataset_str in ['small', 'giant_cora', 'giant_citeseer']:
        return load(dataset_str)
    else:
        data = load_npz_data(dataset_str, seed)
    return data


def load_npz_data(dataset_str, seed):
    with np.load('data/npz/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)
        adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                shape=loader['adj_shape']).tocoo()
        if dataset_str[:2] == 'ms':
            edge_list = torch.cat((torch.tensor(adj_mat.row).type(torch.int64).view(1, -1),
                                   torch.tensor(adj_mat.col).type(torch.int64).view(1, -1)), dim=0)
        else:
            edge_list1 = torch.cat((torch.tensor(adj_mat.row).type(torch.int64).view(1, -1),
                                    torch.tensor(adj_mat.col).type(torch.int64).view(1, -1)), dim=0)
            edge_list2 = torch.cat((torch.tensor(adj_mat.col).type(torch.int64).view(1, -1),
                                    torch.tensor(adj_mat.row).type(torch.int64).view(1, -1)), dim=0)
            edge_list = torch.cat([edge_list1, edge_list2], dim=1)

        edge_list = add_self_loops(edge_list, loader['adj_shape'][0])
        adj = normalize_adj(edge_list)
        if 'attr_data' in loader:
            feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape']).todense()
        elif 'attr_matrix' in loader:
            feature_mat = loader['attr_matrix']
        else:
            feature_mat = None
        features = torch.tensor(feature_mat)

        if 'labels_data' in loader:
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape']).todense()
        elif 'labels' in loader:
            labels = loader['labels']
        else:
            labels = None
        labels = torch.tensor(labels).long()
        train_mask, val_mask, test_mask = split_data(labels, 20, 500, seed)

    data = Data(adj, edge_list, features, labels, train_mask, val_mask, test_mask)

    return data

def load_planetoid_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("./data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.Tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1))
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    edge_list = adj_list_from_dict(graph)
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)
    #adj = edglist2adj(edge_list)

    train_mask = index_to_mask(train_idx, labels.shape[0])
    val_mask = index_to_mask(val_idx, labels.shape[0])
    test_mask = index_to_mask(test_idx, labels.shape[0])

    data = Data(adj, edge_list, features, labels, train_mask, val_mask, test_mask)
    return data


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    # converting 1 undirected edges -> 2 directed edges
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def split_data(labels, n_train_per_class, n_val, seed):
    np.random.seed(seed)
    n_class = int(torch.max(labels)) + 1
    train_idx = np.array([], dtype=np.int64)
    remains = np.array([], dtype=np.int64)
    for c in range(n_class):
        candidate = torch.nonzero(labels == c).T.numpy()[0]
        np.random.shuffle(candidate)
        train_idx = np.concatenate([train_idx, candidate[:n_train_per_class]])
        remains = np.concatenate([remains, candidate[n_train_per_class:]])
    np.random.shuffle(remains)
    val_idx = remains[:n_val]
    test_idx = remains[n_val:]
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask
