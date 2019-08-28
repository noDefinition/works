from collections import Counter, OrderedDict as Od

import numpy as np
from sklearn import metrics

s_nmi = 'nmi'
s_acc = 'acc'
s_ari = 'ari'
eval_scores = (s_acc, s_ari, s_nmi)


def merge(items):
    res = list()
    for a in items:
        res.extend(a)
    return res


def shuffle(array, inplace=False):
    array = array if inplace else array[:]
    np.random.shuffle(array)
    return array


def rehash(items, sort=True):
    items = sorted(set(items)) if sort else set(items)
    return {item: idx for idx, item in enumerate(items)}


def reindex(items):
    item2idx = dict((item, idx) for idx, item in enumerate(sorted(set(items))))
    return [item2idx[item] for item in items]


def count_occurence(y1, y2):
    y1_to_counter = Od((y, Counter()) for y in set(y1))
    for v1, v2 in zip(y1, y2):
        y1_to_counter[v1][v2] += 1
    return y1_to_counter


def score(y_true, y_pred, using_score):
    func = {
        s_acc: ACC,
        s_ari: metrics.adjusted_rand_score,
        s_nmi: NMI,
        # s_nmi: metrics.normalized_mutual_info_score,
        'auc': metrics.roc_auc_score,
    }[using_score.lower()]
    return round(float(func(y_true, y_pred)), 4)


def scores(y_true, y_pred, using_scores=eval_scores):
    return Od((s, score(y_true, y_pred, s)) for s in using_scores)


def NMI(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def ACC(y_true, y_pred):
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    y_true, y_pred = reindex(y_true), reindex(y_pred)
    assert len(y_true) == len(y_pred)
    d = max(max(y_true), max(y_pred)) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for y_t, y_p in zip(y_true, y_pred):
        w[y_t][y_p] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i][j] for i, j in zip(*ind)]) / len(y_pred)


def mean_std(array):
    return np.mean(array), np.std(array, ddof=1)


def permutation_generator(v_max, generator=None):
    if 0 in v_max:
        raise ValueError('0 should not appear in v2max')
    v_num = len(v_max)
    idx_vec = [0] * v_num
    while True:
        yield generator(idx_vec) if generator else idx_vec
        idx = 0
        while idx < v_num and idx_vec[idx] == v_max[idx] - 1:
            idx_vec[idx] = 0
            idx += 1
        if idx < v_num:
            idx_vec[idx] += 1
        else:
            return


def grid_params(name_value_list):
    def vec2dict(idx_vec):
        od = Od(zip(names, [None] * len(names)))
        for n_idx, v_idx in enumerate(idx_vec):
            n, v = names[n_idx], values_list[n_idx][v_idx]
            od[n] = v
        return od

    names, values_list = list(zip(*name_value_list))
    v_len = [len(v) for v in values_list]
    return [od for od in permutation_generator(v_len, vec2dict)]


def cosine_similarity(mtx1, mtx2=None, dense_output=True):
    return metrics.pairwise.cosine_similarity(mtx1, mtx2, dense_output)


def transpose(array):
    # items in array should share the same length
    return list(zip(*array))


def split_slices(array, batch_size):
    for since, until in split_since_until(len(array), batch_size):
        yield array[since: until]


def split_since_until(max_len, batch_size):
    since, until = 0, min(max_len, batch_size)
    while since < max_len:
        yield since, until
        since += batch_size
        until += min(max_len - until, batch_size)


def _can_include(v, include=None, exclude=None):
    if include is not None:
        return v in include
    if exclude is not None:
        return v not in exclude
    return True


def entries2name(entries, include=None, exclude=None, inner='=', inter=',', postfix=''):
    from collections import Iterable
    if isinstance(entries, dict):
        kv_list = entries.items()
    elif isinstance(entries, Iterable):
        kv_list = entries
    else:
        raise TypeError('unexpected type : {}'.format(type(entries)))
    pairs = ['{}{}{}'.format(k, inner, v) for k, v in kv_list if _can_include(k, include, exclude)]
    return inter.join(pairs) + postfix


def name2entries(name, include=None, exclude=None, inner='=', inter=',', postfix=''):
    if not isinstance(name, str):
        raise TypeError('unexpected type : {}'.format(type(name)))
    kv_list = [kv_pair.split(inner) for kv_pair in name.rstrip(postfix).split(inter)]
    entries = [(k, v) for k, v in kv_list if _can_include(k, include, exclude)]
    return entries
