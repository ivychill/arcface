import numpy as np
import torch
from log import logger

# true top1 at false top1
def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=1):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def compute_mAP(qf, q_pids, gf, g_pids):
    # query
    q_camids = np.zeros(qf.shape[0], dtype=np.int)
    # gallery
    g_camids = np.ones(gf.shape[0], dtype=np.int)
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

    return cmc, mAP

def compute_rank1(distmat, max_index, q_pids, g_pids, threshold):
    # index = np.argsort(distmat)  # from small to large
    # max_index = index[:, 0]
    query_num = distmat.shape[0]

    acc = 0
    err = 0
    miss = 0

    for i in range(query_num):
        # logger.debug('distmat[i, max_index[i]]: {}'.format(distmat[i, max_index[i]]))
        if distmat[i, max_index[i]] < threshold:
            if q_pids[i] == g_pids[max_index[i]]:
                acc += 1
            else:
                err += 1
        else:
            miss += 1

    logger.debug('threshold {:.2f}, acc {}, err {}, miss {}'.format(threshold, acc, err, miss))
    acc = acc/float(query_num)
    err = err/float(query_num)
    miss = miss/float(query_num)

    return acc, err, miss
