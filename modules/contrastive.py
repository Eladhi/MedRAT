import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import random
import itertools

class AugmentImage:
   """
   A stochastic data augmentation module
   Transforms any given data example randomly
   resulting in two correlated views of the same example,
   denoted x ̃i and x ̃j, which we consider as a positive pair.
   """

   def __init__(self):
       blur = T.GaussianBlur((3, 3), (0.1, 2.0))

       self.train_transform = torch.nn.Sequential(
           T.Resize(size=256),
           T.RandomCrop(size=224),
           T.RandomAffine(degrees=10, shear=5),
           T.ColorJitter(contrast=0.2, brightness=0.2),
           T.RandomApply([blur], p=0.5),
           # imagenet stats
           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       )

   def __call__(self, x):
       return self.train_transform(x), self.train_transform(x)


class AugmentText:
   """
   A stochastic data augmentation module
   Transforms any given data example randomly
   resulting in two correlated views of the same example,
   denoted x ̃i and x ̃j, which we consider as a positive pair.
   """

   def __init__(self):
       self.pad = 0
       self.eos = 1

   def train_transform(self, x):  # sentence-shuffle
       _x = x.clone()
       for i in range(x.shape[0]):
           if x[i][-1] != self.pad:
               _x[i][-1] = self.eos
               last_valid = (_x[i][1:] == self.eos).nonzero().max() + 1
           else:
                last_valid = (_x[i][1:] == self.pad).nonzero().min()
           sample = x[i][1:last_valid+1]
           sent_split_e, sent_split_b = (sample == self.eos).nonzero(), (sample == self.eos).nonzero().roll(1) + 1
           if sent_split_e.shape[0] == 0:
               continue
           sent_split_b[0] *= 0
           sent_split = torch.concat((sent_split_b, sent_split_e), dim=1)
           sent_split = sent_split[torch.randperm(sent_split.shape[0])]
           w_idx = 1
           for j in range(sent_split.shape[0]):
               w_len = sent_split[j, 1] - sent_split[j, 0] + 1
               _x[i][w_idx:w_idx+w_len] = sample[sent_split[j, 0]:sent_split[j, 1]+1]
               w_idx += w_len
       return _x

   def __call__(self, x):
       return self.train_transform(x), self.train_transform(x)


def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, batch_size, temperature=0.5):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       sim_ij = torch.diag(similarity_matrix, batch_size)
       sim_ji = torch.diag(similarity_matrix, -batch_size)

       positives = torch.cat([sim_ij, sim_ji], dim=0)

       nominator = torch.exp(positives / self.temperature)

       denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
       loss = torch.sum(all_losses) / (2 * self.batch_size)
       return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels.dim() == 2:
            mul = torch.matmul(labels, labels.T)
            mask = mul > 0
            mask = mask.float().to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss