import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from utils import *

class FBMF(nn.Module):
    '''
    full-batch training for matrix factorize
    '''
    def __init__(self, drug_embeddings:torch.Tensor, adr_embeddings:torch.Tensor, lr=0.2) -> None:
        super(FBMF, self).__init__()
        self.drug_embeddings = nn.Parameter(drug_embeddings)
        self.adr_embeddings = nn.Parameter(adr_embeddings)
        self.h = nn.Parameter(torch.empty(1, drug_embeddings.shape[1]))
        nn.init.xavier_normal_(self.h)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self):
         scores = torch.mm((self.h * self.drug_embeddings), self.adr_embeddings.T)
        # scores = self.drug_embeddings * s
         return scores
    
    def fit(self, train_mat, epochs=500, weight=1, lamb=1e-4):
        
        for e in range(epochs):
            self.scores = self.forward()
            loss = F.binary_cross_entropy_with_logits(self.scores, train_mat.float(), reduction='none')
            loss = torch.mean(loss * weight)
            reg = torch.norm(self.drug_embeddings, 2) + torch.norm(self.adr_embeddings, 2) + torch.norm(self.h, 2)
            loss = loss + lamb * reg
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def inference(self, logit=True):
        if logit == True:
            return torch.sigmoid(self.forward().detach())
        else:
            return self.forward().detach()