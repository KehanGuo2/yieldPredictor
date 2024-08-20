import os
import sys
import pickle
import numpy as np
import json, ujson
import pandas as pd
from rdkit import Chem
from typing import List
import sklearn
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import torch_geometric
from dgl import DGLGraph
from rdkit.Chem import rdMolDescriptors as rdDesc
from data import rxn
from tqdm import tqdm
from rxnfp.tokenization import SmilesTokenizer



class data_process():
    def __init__(self, use_graph = True, use_smile = True, use_features = True):
        self.use_graph = use_graph
        self.use_smile = use_smile
        self.use_features = use_features
        self.feature_dim = 7653
        self.mean = 0
        self.std = 0.1
        self.folder = "/data/datav6_internal"
        self.mols_folder = "/data/datav6_internal/molecules"
        self.rxns = rxn(self.folder)
        self.batch_size = 64
        self.Tokenizer = SmilesTokenizer('./pretrained/bert_mlm_1k_tpl/vocab.txt')
        self.x_map = { 'atomic_num':list(range(0, 40)),'chirality': [
                            'CHI_UNSPECIFIED',
                            'CHI_TETRAHEDRAL_CW',
                            'CHI_TETRAHEDRAL_CCW',
                            'CHI_OTHER',
                            'CHI_TETRAHEDRAL',
                            'CHI_ALLENE',
                            'CHI_SQUAREPLANAR',
                            'CHI_TRIGONALBIPYRAMIDAL',
                            'CHI_OCTAHEDRAL',
                        ],'degree':list(range(0, 11)),
                        'formal_charge':list(range(-5, 7)),
                        'num_hs':list(range(0, 9)),
                        'num_radical_electrons':list(range(0, 5)),
                        'hybridization': ['UNSPECIFIED','S','SP','SP2','SP3',
                            'SP3D',
                            'SP3D2',
                            'OTHER',
                        ], 'is_aromatic': [False, True],'is_in_ring': [False, True] }
        self.e_map = { 'bond_type': [
                        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE','ONEANDAHALF',
                        'TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF','AROMATIC','IONIC','HYDROGEN',
                        'THREECENTER','DATIVEONE','DATIVE','DATIVEL','DATIVER','OTHER','ZERO',
                    ],
                    'stereo': [
                        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
                    ],
                    'is_conjugated': [False, True],
                    }


        
    def get_data_spilt(self):
        '''
        data: dict keys are the different data splits.
        '''
        aimnet_des_splits = pickle.load(open(os.path.join(self.folder, "aimnet_des_splits.pkl"), "rb"))
        data = {}
        for key in aimnet_des_splits.keys():
            tmp = {}
            train_test, valid_ids = aimnet_des_splits[key]
            train_ids, test_ids = train_test[:26740],train_test[26740:]
            tmp['train'] = train_ids
            tmp['valid'] = valid_ids
            tmp['test'] = test_ids
            data[key] = tmp
        return data

    def get_labels(self, ids):
        Y = []
        for id in ids:
            y = self.rxns.get_yield(id)
            Y.append(y)
        return np.array(Y)     


    def get_raw_feature(self, ids, dim_reduce=False):
        X = []
        X_fp, X_aimnet, X_mordred, X_aev = [], [], [], []

        for id in tqdm(ids):
            context = self.rxns.get_context_one_hot(id)
            fp = self.rxns.get_fp(id)
            context_fp0 = list(np.array(context + fp)) # 3807
            X_fp.append(context_fp0)
            
            t, T = self.rxns.get_tT(id)
            tT0 = [(t-14.20)/14.38, (T-20.93)/11.13]  #standardize 2
            aimnet0 = self.rxns.get_aimnet_descriptors_(id)  #51
            aimnet = aimnet0 + tT0   
            X_aimnet.append(aimnet)
            
            mordred0 = self.rxns.get_mordred(id)  #4944
            X_mordred.append(mordred0)
            
            aev0 = self.rxns.get_aev_(id) # 5148
            X_aev.append(aev0)
            
            x = aimnet0 + context_fp0 + mordred0 + aev0 + tT0
            X.append(x)
            
        def fea_reduce(x,num):
            print('reduce feature dim')
            pca = PCA(n_components = num)
            pca = pca.fit(x)
            x_new = pca.transform(x)
            return x_new
         
        if dim_reduce==False:
            self.feature_dim = len(X[0])
            return np.array(X)
        else:
            X_fp = fea_reduce(np.array(X_fp),1600)
            X_mordred = fea_reduce(np.array(X_mordred),2000)
            X_aev = fea_reduce(np.array(X_aev),2200)
            X_aimnet = np.array(X_aimnet)
            X = np.hstack((X_fp, X_mordred, X_aev,X_aimnet))
            self.feature_dim = X.shape[1]
            print(self.feature_dim)
            return X
        

    def get_atom_features(self, atom):
        possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'H' , 'S','Si'] #DU代表其他原子

        x = []
        x.append(possible_atom.index(atom.GetSymbol()))
        x.append([-1,0,1,2,3,4,5,6,7,].index(atom.GetImplicitValence()))
        x.append(self.x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(self.x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(self.x_map['degree'].index(atom.GetTotalDegree()))
        x.append(self.x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(self.x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(self.x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(self.x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(self.x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(self.x_map['is_in_ring'].index(atom.IsInRing()))

        return np.array(x) 

    def get_bond_features(self,bond):

        e = []
        e.append(self.e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(self.e_map['stereo'].index(str(bond.GetStereo())))
        e.append(self.e_map['is_conjugated'].index(bond.GetIsConjugated()))

        return np.array(e) 


    def from_smile2graph(self, molecule_smiles):
        G = DGLGraph()
        molecule = Chem.MolFromSmiles(molecule_smiles)
        G.add_nodes(molecule.GetNumAtoms())
        node_features = []
        edge_features = []

        for i in range(molecule.GetNumAtoms()):
            atom_i = molecule.GetAtomWithIdx(i) 
            atom_i_features = self.get_atom_features(atom_i) 
            node_features.append(atom_i_features)

            for j in range(molecule.GetNumAtoms()):
                bond_ij = molecule.GetBondBetweenAtoms(i, j)
                if bond_ij is not None:
                    G.add_edges(i,j) 
                    bond_features_ij = self.get_bond_features(bond_ij) 
                    edge_features.append(bond_features_ij)

        G.ndata['attr'] = torch.from_numpy(np.array(node_features)).type(torch.float32)  #dgl添加原子/节点特征
        G.edata['edge_attr'] = torch.from_numpy(np.array(edge_features)).type(torch.float32) #dgl添加键/边特征
        return G

    def get_2dgraph(self, ids):
        X = []
        for id in tqdm(ids):
            rmols = []
            pmols = []
            tmp = self.rxns.get_smile(id)
            rmols_1 = self.from_smile2graph(tmp[0])
            rmols_2 = self.from_smile2graph(tmp[1])
            x = [rmols_1, rmols_2, self.from_smile2graph(tmp[-1])]
            X.append(x)
        X = np.array(X)
        return X
    
    def get_simes(self, ids):
        X = []
        for id in tqdm(ids):
            tmp = self.rxns.get_smile(id)
            reaction_smiles = tmp[0]+'.'+tmp[1]+'>>'+tmp[-1]
            X.append(reaction_smiles)
            
        X = self.Tokenizer(X, padding = 'max_length', truncation= True, max_length = 320)['input_ids']
        X = np.array(X)
        return X



    
    def load_data(self, data):
        
        def collate_reaction(batch):
            batchdata = list(map(list, zip(*batch)))
            gs = [dgl.batch(s) for s in batchdata[:3]]
            labels = torch.FloatTensor(batchdata[-1])
            smiles = torch.tensor(batchdata[-2])
            xs = torch.FloatTensor(np.array(batchdata[3]))
            gs.append(xs)
            gs.append(smiles)
            return (gs, labels)
        
        train, valid, test = data['train'],data['valid'],data['test']
        Y_train = self.get_labels(train)
        self.mean = Y_train.mean()
        self.std = Y_train.std()
        Y_valid = self.get_labels(valid)
        Y_test = self.get_labels(valid)
        Y_train = (Y_train - self.mean)/self.std
        Y_test = (Y_test - self.mean)/self.std
        Y_valid = (Y_valid - self.mean)/self.std
        all_features = self.get_raw_feature(train+valid+test)
        
        train_feas = all_features[:len(train)]
        valid_feas = all_features[len(train):len(train)+len(valid)]
        test_feas = all_features[len(train)+len(valid):]
        
        train_graph, train_smiles = self.get_2dgraph(train), self.get_simes(train) 
        valid_graph, valid_smiles = self.get_2dgraph(valid), self.get_simes(valid)
        test_graph, test_smiles = self.get_2dgraph(test), self.get_simes(test)
        
        train_loader = DataLoader(my_ds(train_feas, train_graph, train_smiles, Y_train), \
                                  batch_size=self.batch_size, collate_fn=collate_reaction)
        valid_loader = DataLoader(my_ds(valid_feas, valid_graph, valid_smiles, Y_valid), \
                                  batch_size=self.batch_size,collate_fn=collate_reaction)
        test_loader = DataLoader(my_ds(test_feas, test_graph, test_smiles, Y_test), \
                                 batch_size=self.batch_size,collate_fn=collate_reaction)
        
        return train_loader,valid_loader,test_loader
    
    def re_stand_label(self, y):
        return y * self.std + self.mean
    
    
class my_ds(Dataset):
    def __init__(self, X_f, X_g, X_s, Y):
        """
        X:  feature graph smiles
        Y: yields
        """
        self.X_f = X_f
        self.X_g = X_g
        self.X_s = X_s
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        g1 = self.X_g[idx][0]
        g2 = self.X_g[idx][1]
        g3 = self.X_g[idx][2]
        x = self.X_f[idx][:]
        s = self.X_s[idx]

        return  (g1, g2, g3, x, s, self.Y[idx])
    
    
if __name__ == "__main__":
    dataset = data_process()
    data = dataset.get_data_spilt()
    for i in data.keys():
        x, y, z = dataset.load_data(data[i])
    