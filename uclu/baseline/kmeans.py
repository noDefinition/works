from sklearn.cluster import KMeans

import numpy as np
from typing import List

from utils import iu, mu, tu, ru, au
import uclu.data.datasets as udata
import uclu.data.document as udoc


def run_kmeans(out_file, features, labels, kmeans_kwarg, extra_kwarg):
    print(kmeans_kwarg, extra_kwarg)
    x_new = KMeans(**kmeans_kwarg, n_jobs=10).fit_transform(features)
    clusters = np.argmax(x_new, axis=1)
    ret = kmeans_kwarg.copy()
    ret.update(extra_kwarg)
    ret.update(au.scores(labels, clusters))
    iu.dump_array(out_file, [ret], mode='a')
    print('--------->', kmeans_kwarg, extra_kwarg)


def _get_docarr_mean_pooling(docarr: List[udoc.Document], w_embed, dim):
    ret = []
    for doc in docarr:
        title_mean = sum((w_embed[i] for i in doc.title), np.zeros(dim))
        body_mean = sum((w_embed[i] for wints in doc.body for i in wints), np.zeros(dim))
        mean_pool = title_mean + body_mean
        ret.append(mean_pool)
    return ret


def get_mean_pooling_multi(docarr: List[udoc.Document], w_embed, dim):
    docarr_list = au.split_multi_process(docarr, 20)
    args = [(docarr, w_embed, dim) for docarr in docarr_list]
    res_list = mu.multi_process(_get_docarr_mean_pooling, args)
    mean_pooling = np.concatenate(res_list, axis=0)
    return mean_pooling


def get_svd_dims(features, dims: List[int]):
    args = [(features, d, 100) for d in dims]
    res_list = mu.multi_process(ru.fit_svd, args)
    for r in res_list:
        print(r.shape)
    res_list.append(features)
    return res_list


def save_args():
    smp = udata.Sampler(udata.DataSo)
    smp.load_basic()
    smp.load_sparse()
    topics = [d.tag for d in smp.docarr]
    args = []
    for dim in smp.d_obj.support_dims:
        smp.prepare_embedding(dim, 1)
        m = get_mean_pooling_multi(smp.docarr, smp.w_embed, dim)
        extra = {'source': 'mean', 'dim': m.shape[1]}
        args.append((m, topics, extra))
        print(m.shape, extra)
    dims = [32, 64, 128, 256, 300]
    tf = smp.get_feature('tf')
    tf_svds = get_svd_dims(tf, dims)
    for m in tf_svds:
        extra = {'source': 'tf', 'dim': m.shape[1]}
        args.append((m, topics, extra))
        print(m.shape, extra)
    tfidf = smp.get_feature('tfidf')
    tfidf_svds = get_svd_dims(tfidf, dims)
    for m in tfidf_svds:
        extra = {'source': 'tfidf', 'dim': m.shape[1]}
        args.append((m, topics, extra))
        print(m.shape, extra)
    iu.dump_pickle('./kmeans_args.pkl', args)


def main():
    out_file = './kmeans_results.json'
    iu.remove(out_file)
    kmeans_kwarg = {'max_iter': 300}
    args = iu.load_pickle('./kmeans_args.pkl')
    print(len(args), [len(k) for k in args])
    args = [(out_file, features, labels, kmeans_kwarg, extra_kwarg)
            for features, labels, extra_kwarg in args]
    mu.multi_process(run_kmeans, args)


if __name__ == '__main__':
    save_args()
    main()
