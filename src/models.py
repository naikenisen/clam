"""Simplified CLAM models"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn_Net_Gated(nn.Module):
    """Gated Attention Network"""
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        A = self.attention_a(x).mul(self.attention_b(x))
        A = self.attention_c(A)
        return A, x


class CLAM_SB(nn.Module):
    """CLAM Single Branch"""
    def __init__(self, n_classes=2, embed_dim=1024, dropout=0.25, k_sample=8):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        
        # Feature extraction and attention
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_net = Attn_Net_Gated(L=512, D=256, dropout=dropout, n_classes=1)
        
        # Classifiers
        self.bag_classifier = nn.Linear(512, n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(512, 2) for _ in range(n_classes)])
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def forward(self, h, label=None, instance_eval=False):
        h = self.fc(h)
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        # Bag-level prediction
        M = torch.mm(A, h)
        logits = self.bag_classifier(M)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        results = {}
        if instance_eval and label is not None:
            results['instance_loss'] = self._compute_instance_loss(A, h, label)
        
        return logits, Y_prob, Y_hat, A, results

    def _compute_instance_loss(self, A, h, label):
        total_loss = 0.0
        inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
        
        for i in range(self.n_classes):
            if inst_labels[i].item() == 1:
                top_p_ids = torch.topk(A, self.k_sample)[1][-1]
                top_p = torch.index_select(h, dim=0, index=top_p_ids)
                top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
                top_n = torch.index_select(h, dim=0, index=top_n_ids)
                
                p_targets = torch.ones(self.k_sample, device=h.device).long()
                n_targets = torch.zeros(self.k_sample, device=h.device).long()
                
                instances = torch.cat([top_p, top_n], dim=0)
                targets = torch.cat([p_targets, n_targets], dim=0)
                logits = self.instance_classifiers[i](instances)
                total_loss += self.instance_loss_fn(logits, targets)
        
        return total_loss / self.n_classes


class CLAM_MB(nn.Module):
    """CLAM Multi Branch"""
    def __init__(self, n_classes=2, embed_dim=1024, dropout=0.25, k_sample=8):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_net = Attn_Net_Gated(L=512, D=256, dropout=dropout, n_classes=n_classes)
        
        self.bag_classifiers = nn.ModuleList([nn.Linear(512, 1) for _ in range(n_classes)])
        self.instance_classifiers = nn.ModuleList([nn.Linear(512, 2) for _ in range(n_classes)])
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def forward(self, h, label=None, instance_eval=False):
        h = self.fc(h)
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes, device=h.device)
        for c in range(self.n_classes):
            logits[0, c] = self.bag_classifiers[c](M[c])
        
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        results = {}
        if instance_eval and label is not None:
            results['instance_loss'] = self._compute_instance_loss(A, h, label)
        
        return logits, Y_prob, Y_hat, A, results

    def _compute_instance_loss(self, A, h, label):
        total_loss = 0.0
        inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
        
        for i in range(self.n_classes):
            if inst_labels[i].item() == 1:
                top_p_ids = torch.topk(A[i], self.k_sample)[1]
                top_p = torch.index_select(h, dim=0, index=top_p_ids)
                top_n_ids = torch.topk(-A[i], self.k_sample)[1]
                top_n = torch.index_select(h, dim=0, index=top_n_ids)
                
                instances = torch.cat([top_p, top_n], dim=0)
                targets = torch.cat([
                    torch.ones(self.k_sample, device=h.device).long(),
                    torch.zeros(self.k_sample, device=h.device).long()
                ], dim=0)
                
                logits = self.instance_classifiers[i](instances)
                total_loss += self.instance_loss_fn(logits, targets)
        
        return total_loss / self.n_classes


def build_model(model_type='clam_sb', n_classes=2, embed_dim=1024, dropout=0.25):
    """Factory function to build models"""
    if model_type == 'clam_sb':
        return CLAM_SB(n_classes=n_classes, embed_dim=embed_dim, dropout=dropout)
    elif model_type == 'clam_mb':
        return CLAM_MB(n_classes=n_classes, embed_dim=embed_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
