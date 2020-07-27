import torch
import torch.nn.functional as F

# ll: B * N
# logits: B * N
# oh_labels: B * N * K (one-hot)
def compute_min_filtering_loss(outputs, oh_labels):
    logits = outputs.get('logits', None)
    ll = outputs.get('ll', None)

    # with log-likel
    if ll is not None:
        B, K = oh_labels.shape[0], oh_labels.shape[-1]
        bcent = F.binary_cross_entropy_with_logits(
                logits.unsqueeze(-1).repeat(1, 1, K),
                oh_labels, reduction='none').mean(1)
        ll = (ll.unsqueeze(-1) * oh_labels).sum(1) / (oh_labels.sum(1) + 1e-8)
        loss = bcent - ll
        loss[ll==0] = float('inf')
        loss, idx = loss.min(1)
        bidx = loss != float('inf')

        outputs['loss'] = loss[bidx].mean()
        outputs['ll'] = ll[bidx, idx[bidx]].mean()
        outputs['bcent'] = bcent[bidx, idx[bidx]].mean()
    else:
        K = oh_labels.shape[-1]
        bcent = F.binary_cross_entropy_with_logits(
                logits.unsqueeze(-1).repeat(1, 1, K),
                oh_labels, reduction='none').mean(1)
        bcent[oh_labels.sum(1)==0] = float('inf')
        bcent, idx = bcent.min(1)
        bidx = bcent != float('inf')
        bcent = bcent[bidx].mean()
        outputs['loss'] = bcent
        outputs['bcent'] = bcent

def compute_anchored_filtering_loss(outputs, anc_idxs, oh_labels):
    logits = outputs.get('logits', None)
    ll = outputs.get('ll', None)

    B = oh_labels.shape[0]
    labels = oh_labels.argmax(-1)
    anc_labels = labels[torch.arange(B), anc_idxs]
    targets = (labels == anc_labels.unsqueeze(-1)).float()
    bcent = F.binary_cross_entropy_with_logits(logits, targets)
    outputs['bcent'] = bcent

    # with log-likel
    if ll is not None:
        ll = ((ll * targets).sum(1) / (targets.sum(1) + 1e-8)).mean()
        outputs['loss'] = bcent - ll
        outputs['ll'] = ll
    else:
        outputs['loss'] = bcent
        outputs['bcent'] = bcent
