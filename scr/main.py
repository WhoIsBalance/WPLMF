import numpy as np
import torch
from utils import *
import warnings
from param_parser import parameter_parser
from Framework import WPLMF
from metric import TopK_Metric
from GraphEmbed import Node2Vec


args = parameter_parser() 
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    aupr_, f1_, precison_, recall_, mcc_, mr_ = [],[],[],[],[],[]
    topk_precision_, topk_recall_ = [],[]

    for fold in range(5):

        print(f'loading data fold{fold}')
        fpt_path = args.root + '\\' + 'structure.txt'
        data_path = args.root + '\\' + f'{fold}'
        train_set_path = data_path + '\\' + 'train_set.txt'
        vaild_set_path = data_path + '\\' + 'valid_set.txt'
        test_set_path = data_path + '\\' + 'test_set.txt'
        kg_path = args.root + '\\' + 'kg.txt'
        save_root = args.save_path + '\\' + f'{fold}'
        train_set = np.loadtxt(train_set_path, dtype=np.int32)
        valid_set = np.loadtxt(vaild_set_path, dtype=np.int32)
        test_set = np.loadtxt(test_set_path, dtype=np.int32)

        kg = np.loadtxt(kg_path, dtype=np.int32)
        kg[:,1] += args.n_drug
        drug_se_mat = np.loadtxt(args.root + '\\' + 'drug_se_mat.txt')
        train_mat = create_matrix(train_set)

        G = create_graph(kg)
        mkdir(save_root)    # 创建结果保存文件夹
        is_train = args.train_kg
        drug_embeddings, adr_embeddings = [],[]
        for t in range(args.n_basemodel):
            vec_path = save_root + '\\' + f'vec{fold}_{t}_fd.txt'
            p = args.pq[t][0]
            q = args.pq[t][1]
            if is_train:
                node2vec = Node2Vec(G, p=p, q=q, n_walks=args.n_walks, walk_length=args.walk_length)
                node2vec._run()
                skipgram_model = node2vec.fit(window=3, min_count=1)
                skipgram_model.wv.save_word2vec_format(vec_path ,binary=False, write_header=False)
                embeddings = load_vec(vec_path)
            else:
                embeddings = load_vec(vec_path)
            embeddings = torch.from_numpy(embeddings).float()

            drug_embeddings.append(embeddings[0:args.n_drug,:].float())
            adr_embeddings.append(torch.randn(size=(args.n_adr, 12)).float())


        model = WPLMF(train_mat, drug_embeddings, adr_embeddings, valid_set, thres=args.thres)
        scores = model.fit(tune=True)
        thres = find_thres(valid_set, scores)
        topk_metric = TopK_Metric(test_set, scores, thres, k=15, n_drug=args.n_drug, n_adr = args.n_adr)
        aupr, f1, prec, recall, mcc, mr, scores, scores_ = eval(test_set, scores, thres, fold) 
        topk_precision = topk_metric.topk_macro_precison()
        topk_recall = topk_metric.topk_macro_recall()
        
        aupr_.append(aupr)
        f1_.append(f1)
        precison_.append(prec)
        recall_.append(recall)
        mcc_.append(mcc)
        mr_.append(mr)
        topk_precision_.append(topk_precision)
        topk_recall_.append(topk_recall)

        print(f'aupr: {aupr}   f1: {f1}   topk_precision:{topk_precision}   topk_recall:{topk_recall}')
        print(thres)
    

    aupr = np.around(np.mean(aupr_), 4)
    f1 = np.around(np.mean(f1_), 4)
    precison = np.around(np.mean(precison_), 4)
    recall = np.around(np.mean(recall_), 4)
    mcc = np.around(np.mean(mcc_), 4)
    mr = np.around(np.mean(mr_), 4)
    topk_precision = np.mean(topk_precision_)
    topk_recall = np.mean(topk_recall_)


    print(f'aupr: {aupr}, {np.std(aupr_)}')
    print(f'f1: {f1}, {np.std(f1_)}')
    print(f'precision: {precison}, {np.std(precison_)}')
    print(f'recall: {recall}, {np.std(recall_)}')
    print(f'mcc: {mcc}, {np.std(mcc_)}')
    print(f'mr: {mr}, {np.std(mr_)}')
    print(f'topk_recall:{topk_recall}, {np.std(topk_precision_)}')
    print(f'topk_precision:{topk_precision}, {np.std(topk_recall_)}')