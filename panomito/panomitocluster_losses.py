import torch
import torch.nn as nn

def reconstruction_loss(x, x_hat, y, y_hat):

    loss_x = nn.MSELoss()(x_hat, x)
    loss_y = nn.MSELoss()(y_hat, y)
    return loss_x + loss_y

def triplet_loss(z, label_x, label_y, margin):

    trip_loss_x = batch_hard_triplet_loss(label_x, z, margin)
    trip_loss_y = batch_hard_triplet_loss(label_y, z, margin)
    return trip_loss_x + trip_loss_y, trip_loss_x, trip_loss_y

def _pairwise_distances(embeddings, squared=False):

    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)

    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)

    distances = torch.clamp(distances, min=0.0) 

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0)).bool().to(labels)
    indices_not_equal = ~indices_equal
    labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    mask = indices_not_equal & labels_equal
    return mask

def _get_anchor_negative_triplet_mask(labels):

    labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    mask = ~labels_equal
    return mask


def _get_triplet_mask(labels):
    indices_equal = torch.eye(labels.size(0)).bool().to(labels)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k
    label_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    valid_labels = i_equal_j & ~i_equal_k
    mask = distinct_indices & valid_labels
    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(labels).float()
    mask = mask.to(embeddings.device)
    triplet_loss = mask * triplet_loss
    triplet_loss = torch.clamp(triplet_loss, min=0.0)
    num_positive_triplets = torch.sum(triplet_loss > 1e-16)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float().to(pairwise_dist.device)
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float().to(pairwise_dist.device)
    max_anchor_negative_dist = torch.max(pairwise_dist, dim=1, keepdim=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)
    triplet_loss = torch.mean(triplet_loss)
    return triplet_loss

if __name__ == "__main__":
    embeddings = torch.randn(8, 1000)
    pairwise_dist = _pairwise_distances(embeddings)