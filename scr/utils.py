import numpy as np
import json
import networkx as nx
from param_parser import parameter_parser
import os
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_recall_curve,auc,precision_score,recall_score,matthews_corrcoef,roc_curve
import torch
from rdkit import Chem
from sklearn.decomposition import KernelPCA
from rdkit.Chem import AllChem

args = parameter_parser()

def create_fpts_matrix(path):

    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    mat = []
    for line in lines:
        pieces = line.strip().split('\t')
        fpt = pieces[2]
        mat.append(list(fpt))
    
    mat = np.array(mat, dtype=np.int32)
    return mat

def create_graph(kg):

    G = nx.Graph()
    kg = kg.tolist()
    for node in range(args.n_entity):
        G.add_node(node)
    for line in kg:
        G.add_edges_from([(line[0],line[1])])

    return G

def create_matrix(pairs):

    x, y, lables = pairs[:,0], pairs[:,1], pairs[:,2]
    matirx = np.zeros((args.n_drug, args.n_adr), dtype=np.int32)
    matirx[x, y ] = lables

    return matirx



def mkdir(path):

    if not os.path.exists(path):
        os.mkdir(path)

def load_vec(path):

    vec = np.loadtxt(path)
    # vec = vec[np.argsort(vec[:,0])][:,1:]
    vec = vec[np.argsort(vec[:,0])]
    return vec[:,1:]


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def find_thres(valid_set, pred_mat, metric='f1'):

    def f1(p, r):
        if p + r != 0:
            return 2 * (p * r) / (p + r)
        else:
            return 0
    def gmean(tpr, fpr):
        return np.sqrt(tpr * (1 - fpr))
    
    def youden(tpr, fpr):
        return tpr - fpr

    drug_indices = valid_set[:,0].astype(int)
    adr_indices = valid_set[:,1].astype(int)
    labels = valid_set[:,2]
    scores = pred_mat[drug_indices, adr_indices]

    # f1s=[f1(p, r) for p, r in zip(precisions, recalls)]
    if metric == 'f1':
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s=[f1(p, r) for p, r in zip(precisions, recalls)]
        optimal_index = np.argmax(f1s)
        
    elif metric == 'gmean':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        gmeans = gmean(tpr, fpr)
        optimal_index = np.argmax(gmeans)

    elif metric == 'youden':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        youdens = youden(tpr, fpr)
        optimal_index = np.argmax(youdens)
        
    optimal_threshold = thresholds[optimal_index]

    return optimal_threshold




def load_sturcture(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        contents = line.strip().split('\t')
        data.append(list(contents[2]))
    data = np.array(data, dtype=np.int32)

    return data


def eval(test_set, pred_mat, thres, t):

    drug_indices = test_set[:,0]
    adr_indices = test_set[:,1]
    labels = test_set[:,2]

    scores = pred_mat[drug_indices, adr_indices]
    scores_ = scores.copy()
    scores_ = (scores_ >= thres).astype('int')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    aupr = auc(recall,precision)
    f1 = f1_score(labels, scores_)
    prec = precision_score(y_true=labels,y_pred=scores_)
    recall = recall_score(y_true=labels,y_pred=scores_)
    mcc = matthews_corrcoef(y_true=labels,y_pred=scores_)
    mr = mrank(labels, scores)

    return aupr, f1, prec, recall, mcc, mr, scores, scores_




def mrank(y, y_pre):
    index = np.argsort(-y_pre)
    r_label = y[index]
    r_index = np.array(np.where(r_label == 1)) + 1
    reci_sum = np.sum(1 / r_index)
    # reci_rank = np.mean(1 / r_index)
    # mr = np.mean(r_index)
    return reci_sum



class EarlyStopping():

    def __init__(self, patience=3, delta=0, mode='g') -> None:
        
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_result = {'pred':[], 'score':[], 'thres':[], 'train':[]}
        self.mode = mode
        self.counter = 0

    def __call__(self, score, pred, thres, train_mats):
        
        if self.mode == 'g':
            if self.best_score is None:
                self.best_score = score
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    idx = self.best_result['score'].index(max(self.best_result['score']))
                    return self.best_result['pred'][idx], self.best_result['thres'][idx], self.best_result['train'][idx]
            else:
                self.best_score = score
                self.counter = 0
                self.best_result = {'pred':[], 'score':[], 'thres':[], 'train':[]}
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
        else:
            if self.best_score is None:
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
                self.best_score = score
            elif score > self.best_score + self.delta:
                self.counter += 1
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    idx = self.best_result['score'].index(max(self.best_result['score']))
                    return self.best_result['pred'][idx], self.best_result['thres'][idx], self.best_result['train'][idx]
            else:
                self.best_score = score
                self.counter = 0
                self.best_result = {'pred':[], 'score':[], 'thres':[], 'train':[]}
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
    
        return pred, thres, train_mats
    


def getTanimotocoefficient(s,t):
    # 计算谷本系数
    # s=np.asarray(s)
    # t=np.asarray(t)
    if (s.shape!=t.shape):
        print("向量长度不一致")
        return np.nan
    return (np.sum(s*t))/(np.sum(s**2)+np.sum(t**2)-np.sum(s*t))

def get_sim_mat(data) -> np.ndarray:
    sim_mat = np.zeros((len(data), len(data)))
    for i in tqdm(range(len(data))):
        fp1 = data[i]
        for j in range(i, len(data)):
            fp2 = data[j]
            sim = getTanimotocoefficient(fp1, fp2)
            sim_mat[i,j] = sim
            sim_mat[j,i] = sim
    sim_mat = np.nan_to_num(sim_mat)

    return sim_mat

def load_sturcture(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        contents = line.strip().split('\t')
        data.append(contents[1])
    # data = np.array(data, dtype=np.int32)

    return data


def load_fpt_vec(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        contents = line.strip().split('\t')
        data.append(contents[1])
    mols = [Chem.MolFromSmiles(canonical_smiles) for canonical_smiles in data]
    data_fps = [list(AllChem.GetMorganFingerprintAsBitVect(x,3,2048).ToBitString()) for x in mols]  # 2048维ECFP摩根指纹
    X = np.array(data_fps, dtype=np.int32)
    scikit_kpca = KernelPCA(n_components=100, kernel='rbf', gamma=15)
    desc = scikit_kpca.fit_transform(X)

    return desc


def predict_topk(k, save_path, drug_se_mat, scores):

    adr2id_path = r'D:\Document\Paper1\dataset\final_dataset1\id2adr.json'
    drug2id_path = r'D:\Document\Paper1\dataset\final_dataset1\id2drug.json'
    f_adr = open(adr2id_path, 'r')
    f_drug = open(drug2id_path, 'r')
    id2adr = json.load(f_adr)
    id2drug = json.load(f_drug)
    f_adr.close()
    f_drug.close()


    drug_se_mat = drug_se_mat
    pred_mat = scores

    neg_mask = torch.abs(drug_se_mat - 1)
    pred_mat = (pred_mat * neg_mask).flatten()
    freq, sorted_indices = torch.sort(pred_mat, descending=True)

    sorted_indices = sorted_indices.numpy()[0:k]
    lables = np.zeros_like(sorted_indices)
    mat_indices = idx2pair(sorted_indices, lables)
    drug_indices = mat_indices[:,0]
    adr_indices = mat_indices[:,1]

    f = open(save_path, 'w')
    for i in range(k):
        drug = id2drug[str(drug_indices[i])]
        adr = id2adr[str(adr_indices[i])]
        line = drug + '\t' + adr + '\n'
        f.write(line)
    f.close()


def idx2pair(indices, labels):

    y_indices = indices % args.n_adr
    x_indices = (indices / args.n_adr).astype(int)
    out = np.stack([x_indices, y_indices, labels], axis=0)

    return out.T.astype(int)


def add_sparsity(train_mat, r:float):

    indices = np.where(train_mat == 1)
    mask = np.zeros_like(train_mat)
    selected = np.random.choice([i for i in range(len(indices[0]))], size=int(len(indices[0]) * r), replace=False)
    x = indices[0][selected]
    y = indices[1][selected]
    mask[x,y] = 1
    train_mat = mask * train_mat
    return train_mat