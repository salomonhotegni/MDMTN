import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(model, outputs, targets, criterion, shared_feats, feats_single, lmbd, device):
    
        num_tasks = len(outputs)
        w = torch.ones(num_tasks).to(device)
        train_loss = []
        for i in range(num_tasks):
            loss_ti = criterion(outputs[i], targets[:, i]).to(device)
            train_loss.append(loss_ti)
        loss = torch.mean(sum(w[i] * train_loss[i] for i in range(num_tasks)))
        
        dist_loss = []
        for i in range(num_tasks):
            feat_ti = feats_single[i].detach()
            feat_ti = feat_ti / torch.norm(feat_ti, dim=1, keepdim=True)
            # print("shared_feats.shape = ", shared_feats.shape)
            feat_si = model.adaptors[i](shared_feats)
            feat_si = feat_si.detach()
            feat_si = feat_si / torch.norm(feat_si, dim=1, keepdim=True)
            # print("feat_si.shape = ", feat_si.shape)
            # print("feat_ti.shape = ", feat_ti.shape)
            dist_i = torch.mean(torch.norm(feat_si - feat_ti, dim=1) ** 2 )
            dist_loss.append(dist_i)

        dist_loss = sum(dist_loss[i] * lmbd[i] for i in range(num_tasks))

        loss = loss + dist_loss
        
        return loss
    
