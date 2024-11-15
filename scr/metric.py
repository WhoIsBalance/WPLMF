from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, auc, precision_score, recall_score
import numpy as np


class TopK_Metric():

    def __init__(self, test_set, scores, thres, k, n_drug, n_adr) -> None:
        
        self.test_set = test_set
        self.thres = thres
        self.scores = scores
        self.k = k
        self.n_drug = n_drug
        self.n_adr = n_adr
        self.top_k_micro()
        self.top_k_macro()

    def top_k_micro(self):

        drug_idx = self.test_set[:,0]
        adr_idx = self.test_set[:,1]
        labels = self.test_set[:,2]
        scores = self.scores[drug_idx, adr_idx]
        idx = np.argsort(scores)[::-1]
        if isinstance(self.k, float):
            self.k = int(len(idx) * self.k)
        topk_idx = idx[0:self.k]
        self.topk_micro_scores = scores[topk_idx]
        self.topk_micro_labels = labels[topk_idx]
        self.topk_micro_preditions = (self.topk_micro_scores > self.thres).astype(int)


    def top_k_macro(self):


        drug_idx = self.test_set[:,0]
        adr_idx = self.test_set[:,1]
        labels = self.test_set[:,2]
        scores = self.scores[drug_idx, adr_idx]
        preditions = (scores > self.thres).astype(int)
        scores_mat = np.zeros((self.n_drug, self.n_adr))
        labels_mat = np.zeros((self.n_drug, self.n_adr))
        preditions_mat = np.zeros((self.n_drug, self.n_adr))
        scores_mat[drug_idx, adr_idx] = scores
        labels_mat[drug_idx, adr_idx] = labels
        preditions_mat[drug_idx, adr_idx] = preditions
        idx = np.argsort(scores_mat, axis=1)[::,::-1]
        if isinstance(self.k, float):
            self.k = int(len(idx) * self.k)
        topk_idx = idx[:,0:self.k]
        sorted_scores = []
        sorted_labels = []
        sorted_predtions = []
        for i in range(topk_idx.shape[0]):
            indices = topk_idx[i]
            score = scores_mat[i]
            label = labels_mat[i]
            predition = preditions_mat[i]
            sorted_scores.append(score[indices])
            sorted_labels.append(label[indices])
            sorted_predtions.append(predition[indices])
        
        self.topk_macro_scores = np.stack(sorted_scores)
        self.topk_macro_labels = np.stack(sorted_labels)
        self.topk_macro_preditions = np.stack(sorted_predtions)


    
    def topk_micro_aupr(self):

        precision, recall, thresholds = precision_recall_curve(self.topk_micro_labels, self.topk_micro_scores)
        aupr = auc(recall,precision)
        return aupr

    def topk_micro_acc(self): return accuracy_score(self.topk_micro_labels, self.topk_micro_preditions)

    def topk_macro_acc(self): 

        return np.mean((self.topk_macro_scores.shape[1] - np.sum(np.abs(self.topk_macro_preditions - self.topk_macro_labels), axis=1)) / self.topk_macro_scores.shape[1])
    
    def topk_macro_precison(self):

        precisons = []
        for i in range(self.topk_macro_labels.shape[0]):
            labels = self.topk_macro_labels[i]
            preditions = self.topk_macro_preditions[i]
            precisons.append(precision_score(labels, preditions))
        return np.mean(precisons)
    
    
    def topk_macro_recall(self):

        recalls = []
        for i in range(self.topk_macro_labels.shape[0]):
            labels = self.topk_macro_labels[i]
            preditions = self.topk_macro_preditions[i]
            recalls.append(recall_score(labels, preditions))
        return np.mean(recalls)