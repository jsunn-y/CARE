import torch
import torch.nn as nn
import torch.nn.functional as F

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.CL_loss == 'EBM_NCE':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.CL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc

def SupConLoss(model_emb, temp, n_pos):
    '''
    return the SupCon-Hard loss
    features:  
        model output embedding, dimension [bsz, n_all, out_dim], 
        where bsz is batchsize, 
        n_all is anchor, pos, neg (n_all = 1 + n_pos + n_neg)
        and out_dim is embedding dimension
    temp:
        temperature     
    n_pos:
        number of positive examples per anchor
    '''
    # l2 normalize every embedding
    features = F.normalize(model_emb, dim=-1, p=2)
    # features_T is [bsz, outdim, n_all], for performing batch dot product
    features_T = torch.transpose(features, 1, 2)
    # anchor is the first embedding 
    anchor = features[:, 0]
    # anchor is the first embedding 
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T)/temp 
    # anchor_dot_features now [bsz, n_all], contains 
    anchor_dot_features = anchor_dot_features.squeeze(1)
    # deduct by max logits, which will be 1/temp since features are L2 normalized 
    logits = anchor_dot_features - 1/temp
    # the exp(z_i dot z_a) excludes the dot product between itself
    # exp_logits is of size [bsz, n_pos+n_neg]
    exp_logits = torch.exp(logits[:, 1:])
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1)) # size [bsz], scale by n_pos
    pos_logits_sum = logits[:, 1:n_pos+1].sum(1) #sum over all (anchor dot pos)
    log_prob = (pos_logits_sum - exp_logits_sum)/n_pos
    loss = - log_prob.mean()
    return loss  