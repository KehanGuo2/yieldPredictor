import os
import pickle
import numpy as np
from rdkit import Chem
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from data import rxn
import dgl
from dgl import DGLGraph
from rdkit import Chem
from dgl.nn.pytorch import NNConv, Set2Set
import random 
import itertools
import wandb
import pdb

x_map = {
    'atomic_num':
    list(range(0, 54)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

def one_of_k_encoding_unk(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'H' , 'S','Si','B']
    
    x = []
    x.append(possible_atom.index(atom.GetSymbol()))
    x.append([-1,0,1,2,3,4,5,6,7,].index(atom.GetImplicitValence()))
    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
    x.append(x_map['degree'].index(atom.GetTotalDegree()))
    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
    x.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
    x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
    
    return np.array(x) 

def get_bond_features(bond):
    
    e = []
    e.append(e_map['bond_type'].index(str(bond.GetBondType())))
    e.append(e_map['stereo'].index(str(bond.GetStereo())))
    e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
    
    return np.array(e) 


def from_smile2graph(molecule_smiles):
    G = DGLGraph()
    molecule = Chem.MolFromSmiles(molecule_smiles)
    G.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []

    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i) 
        atom_i_features = get_atom_features(atom_i) 
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i,j) 
                bond_features_ij = get_bond_features(bond_ij) 
                edge_features.append(bond_features_ij)

    G.ndata['attr'] = torch.from_numpy(np.array(node_features)).type(torch.float32)  #dgl add node feature
    G.edata['edge_attr'] = torch.from_numpy(np.array(edge_features)).type(torch.float32) #dgl edge feature
    return G


class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1024):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats
    
    
class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 readout_feats = 1024,
                 predict_hidden_feats = 512,output_size = 2, prob_dropout = 0.2):
        
        super(reactionMPNN, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats)

        self.predict = nn.Sequential(
            nn.Linear(readout_feats * 2, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, output_size)
        )
    
    def forward(self, x):
        rmols = x[:2]
        pmols = x[2]
        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = self.mpnn(pmols) #torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)

        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out = self.predict(concat_feats)
        return out[:,0], out[:,1]

def collate_reaction_graphs(batch):

    batchdata = list(map(list, zip(*batch)))
    gs = [dgl.batch(s) for s in batchdata[:-1]]
    labels = torch.FloatTensor(batchdata[-1])
    return (gs, labels)

def get_raw_feature(ids,standardize=True):
    X, Y = [], []
    for id in tqdm(ids):
        rmols = []
        pmols = []
        tmp = rxns.get_smile(id)
        #yield_y = rxns.get_yield(id)
        rmols_1 = from_smile2graph(tmp[0])
        rmols_2 = from_smile2graph(tmp[1])
        
        x = [rmols_1, rmols_2, from_smile2graph(tmp[-1])]
        X.append(x)
        y = rxns.get_yield(id)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

def get_data_spilt():
    with open('/afs/crc.nd.edu/user/k/kguo2/uncertainty_yld/data/datav6_internal/normal_ids.pkl', 'rb') as file:
        normal_ids = pickle.load(file)
    with open('/afs/crc.nd.edu/user/k/kguo2/uncertainty_yld/data/datav6_internal/train_uncertain_ids.pkl', 'rb') as file:
        uncertain_ids = pickle.load(file)
    with open('/afs/crc.nd.edu/user/k/kguo2/uncertainty_yld/data/datav6_internal/test_clean_ids.pkl', 'rb') as file:
        test_ids = pickle.load(file)
    with open('/afs/crc.nd.edu/user/k/kguo2/uncertainty_yld/data/datav6_internal/test_uncertain_ids.pkl', 'rb') as file:
        test_u_ids = pickle.load(file)
    
    tmp = {}
    tmp['normal'] = normal_ids
    tmp['uncertain'] = uncertain_ids
    tmp['test'] = test_ids
    tmp['test_u'] = test_u_ids
    
    return tmp


def load_data(data, get_feature,config):

    normal,uncertain,test,test_u = data['normal'],data['uncertain'],data['test'],data['test_u']
    # combine normal and uncertain and split to train and valid and cali
    data = normal

    test_u = list(test_u)
    test = list(test)
    # test = list(test) + test_u[:int(len(test_u)*0.2)]
    test_u = test_u[int(len(test_u)*0.2):]

    train_ids = set(random.sample(list(data), int(len(data) * 0.9)))
    val_ids = data - train_ids
    
    X_train, Y_train = get_feature(train_ids)
    global mean
    mean = 64.05
    global std
    std = 22.46
    X_valid,Y_valid = get_feature(val_ids)
    # X_cal,Y_cal = get_feature(cali)
    X_test,Y_test = get_feature(test)
    X_test_u,Y_test_u = get_feature(test_u)
    # X_test_u,Y_test_u = get_feature(test_u)
    Y_train = (Y_train - mean)/std
    Y_test = (Y_test - mean)/std
    # Y_cal = (Y_cal - mean)/std
    Y_test_u = (Y_test_u - mean)/std
    Y_valid = (Y_valid - mean)/std
    train_loader = DataLoader(my_ds(X_train, Y_train), batch_size=config.batch_size,collate_fn=collate_reaction_graphs,shuffle=True)
    valid_loader = DataLoader(my_ds(X_valid, Y_valid), batch_size=config.batch_size, collate_fn=collate_reaction_graphs,shuffle=True)
    # cal_loader = DataLoader(my_ds(X_cal, Y_cal), batch_size=len(X_cal))
    test_loader = DataLoader(my_ds(X_test, Y_test), batch_size=config.batch_size, collate_fn=collate_reaction_graphs,shuffle=True)
    test_loader_u = DataLoader(my_ds(X_test_u, Y_test_u), batch_size=config.batch_size, collate_fn=collate_reaction_graphs,shuffle=True)
    return train_loader,valid_loader,test_loader,test_loader_u

def train(model,data_loader,device,optimizer):
    model.to(device)
    model.train()
    running_loss = 0.0
    # try different loss.
    criterion = nn.L1Loss()
    for xs, ys in data_loader:
        xs = [i.to(device) for i in xs]
        ys = ys.to(device).reshape(-1, 1)
        out,logvar = model(xs)
        loss = criterion(out, ys.view(-1, 1))
        loss = (1 - 0.1) * loss.mean() + 0.1 * ( loss * torch.exp(-logvar) + logvar ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)  
    train_r2, train_mae, train_rmse = valid(model,data_loader,device)
    return (train_r2, train_mae, train_rmse,out,epoch_loss)

def valid(model, data_loader,device):
    model.eval()
    Ys, Y_hats = [], []
    with torch.no_grad():
        for xs, ys in data_loader:
            xs = [i.to(device) for i in xs]
            ys = ys.reshape(-1, 1).numpy()
            out,_ = model(xs)
            Ys.append(ys)
            Y_hats.append(out.detach().to("cpu").numpy())
    Ys = np.concatenate(Ys, axis=0) * std + mean
    Y_hats = np.concatenate(Y_hats, axis=0) * std + mean
    r2 = r2_score(Ys, Y_hats)
    mae = mean_absolute_error(Ys, Y_hats)
    rmse = mean_squared_error(Ys, Y_hats, squared=False)
    return (r2, mae, rmse)

def test(model,checkpt_file,data_loader,device):
    model.load_state_dict(torch.load(checkpt_file))
    model.to(device)
    return valid(model,data_loader,device)

class my_ds(Dataset):
    def __init__(self, X, Y):
        """
        X: Total feature length 2705 aimnet 50 context_fp 324 mordred 1295
        Y: yields"""
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        g1 = self.X[idx][0]
        g2 = self.X[idx][1]
        g3 = self.X[idx][2]

        return  (g1, g2, g3, self.Y[idx])
    
reaxys_mean = 64.05
reaxys_median = 68.00
reaxys_std = 22.46
# preparing data
global mean
global std
mean = reaxys_mean
std = reaxys_std

folder = "/afs/crc.nd.edu/user/k/kguo2/uncertainty_yld/data/datav6_internal/"
rxns = rxn(folder)
ids = rxns.all_idx2()
print(f'Total number of reactions: {len(ids)}')

train_ids0 = pickle.load(open(os.path.join(folder, 'normal_ids.pkl'), 'rb'))
train_uncertain_ids = pickle.load(open(os.path.join(folder, 'train_uncertain_ids.pkl'),'rb'))
# train_ids0 = train_ids0.union(train_uncertain_ids)
test_ids = pickle.load(open(os.path.join(folder, 'test_clean_ids.pkl'), 'rb'))
test_ids_u = pickle.load(open(os.path.join(folder, 'test_uncertain_ids.pkl'), 'rb'))


hyperparameter_grid = {
    "output_size": [2], 
    "node_in_feature": [11],
    "edge_in_feature": [3],
    "readout_feats": [1024],
    "predict_hidden_feats": [512],
    "learning_rate": [0.001],
    "epochs": [200] ,#try other epoch nums,
    "batch_size": [128],
    "checkpt_path": ["./final_models"], #modify to the path you want to save the final model,
    "dropout": [0.2]
}

for config in itertools.product(*hyperparameter_grid.values()):
    config_dict = dict(zip(hyperparameter_grid.keys(), config))
    for i in range(5):
        mean_mse = []
        mean_mae = []
        mean_r2 = []
        mean_r2_u = []
        mean_mae_u = []
        mean_mse_u = []

    # Initialize wandb with the current configuration, replace with your project name
        wandb.init(project="gnn_baseline", entity="nd", config=config_dict)
        device = torch.device("cuda:0")
        config = wandb.config
        all_data = get_data_spilt()
        train_loader, valid_loader,test_loader,test_loader_u = load_data(all_data,get_raw_feature,config)

        model = reactionMPNN(config.node_in_feature,config.edge_in_feature,config.readout_feats,\
                config.predict_hidden_feats,config.output_size,config.dropout)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        bad_counter = 0
        best = 8848
        for epoch in range(config.epochs):
            train_r2, train_mae, train_rmse,pred,epoch_loss = train(model,train_loader,device,optimizer)
            wandb.log({"train_r2": train_r2, "train_mae": train_mae, "train_rmse": train_rmse, "epoch_loss": epoch_loss})
            valid_r2, valid_mae, valid_rmse = valid(model,valid_loader,device)
            wandb.log({"valid_r2": valid_r2, "valid_mae": valid_mae,"valid_rmse": valid_rmse})
            
            if valid_rmse < best:
                best = valid_rmse
                bad_counter = 0
                torch.save(model.state_dict(), os.path.join(config.checkpt_path, "best_model.pt"))
            else:
                bad_counter += 1
        

        model = reactionMPNN(config.node_in_feature,config.edge_in_feature,config.readout_feats,\
                config.predict_hidden_feats,config.output_size,config.dropout)
        model.load_state_dict(torch.load(os.path.join(config.checkpt_path, "best_model.pt")))
        model.to(device)
        test_r2, test_mae, test_rmse = valid(model,test_loader,device)
        test_r2_u, test_mae_u, test_rmse_u = valid(model,test_loader_u,device)

        print(f"Test R2: {test_r2:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Test Uncertain R2: {test_r2_u:.4f}, Test Uncertain MAE: {test_mae_u:.4f}, Test Uncertain RMSE: {test_rmse_u:.4f}")
        mean_r2.append(test_r2)
        mean_mae.append(test_mae)
        mean_mse.append(test_rmse)

        mean_r2_u.append(test_r2_u)
        mean_mae_u.append(test_mae_u)
        mean_mse_u.append(test_rmse_u)

        wandb.log({"test_r2": test_r2, "test_mae": test_mae, "test_rmse": test_rmse})
    wandb.log({"mean_test_r2": np.mean(mean_r2), "mean_test_mae": np.mean(mean_mae), "mean_test_rmse": np.mean(mean_mse)})
    wandb.log({"mean_test_r2_u": np.mean(mean_r2_u), "mean_test_mae_u": np.mean(mean_mae_u), "mean_test_rmse_u": np.mean(mean_mse_u)})
    wandb.finish()
