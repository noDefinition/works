from typing import List
import numpy as np
from sklearn.cluster import KMeans

from utils import iu, mu, ru, au
import uclu.data.datasets as udata
import uclu.data.document as udoc


def run_kmeans(out_file, features, labels, kmeans_kwarg, extra_kwarg):
    print(kmeans_kwarg, extra_kwarg)
    x_new = KMeans(**kmeans_kwarg, n_jobs=10).fit_transform(features)
    clusters = np.argmax(x_new, axis=1)
    kmeans_kwarg.update(extra_kwarg)
    kmeans_kwarg.update(au.scores(labels, clusters))
    iu.dump_array(out_file, [kmeans_kwarg], mode='a')
    print('---- over ---->', kmeans_kwarg, extra_kwarg)


def _get_docarr_mean_pooling(docarr: List[udoc.Document], w_embed, add_body):
    ret = []
    w_num, w_dim = w_embed.shape
    for doc in docarr:
        title_mean = sum((w_embed[i] for i in doc.title), np.zeros(w_dim)) / len(doc.title)
        if add_body:
            body_sum = sum((w_embed[i] for wints in doc.body for i in wints), np.zeros(w_dim))
            body_mean = body_sum / sum(len(wints) for wints in doc.body)
            mean_pool = title_mean + body_mean
        else:
            mean_pool = title_mean
        ret.append(mean_pool)
    return ret


def get_mean_pooling_multi(docarr: List[udoc.Document], w_embed, add_body):
    docarr_list = mu.split_multi(docarr, 20)
    args = [(docarr, w_embed, add_body) for docarr in docarr_list]
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


def save_args(add_body: bool):
    smp = udata.Sampler(udata.DataSo)
    smp.load_basic()
    smp.fit_sparse(add_body=add_body, tfidf=True, p_num=20)
    topics = [d.tag for d in smp.docarr]

    args = []
    for dim in smp.d_obj.support_dims:
        smp.prepare_embedding(dim=dim, topic_ratio=1)
        m = get_mean_pooling_multi(smp.docarr, smp.w_embed, add_body)
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

    iu.dump_pickle(get_args_file(add_body), args)


def main(add_body: bool):
    postfix = 'add_body' if add_body else 'no_body'
    log_file = './kmeans_results_{}.json'.format(postfix)
    iu.remove(log_file)

    kmeans_kwarg = {'max_iter': 300}
    args = iu.load_pickle(get_args_file(add_body))
    print(len(args), [len(k) for k in args])
    args = [(log_file, features, labels, kmeans_kwarg, extra_kwarg)
            for features, labels, extra_kwarg in args]
    mu.multi_process(run_kmeans, args)


def get_args_file(add_body: bool):
    postfix = 'add_body' if add_body else 'no_body'
    args_file = './kmeans_args_{}.pkl'.format(postfix)
    return args_file


if __name__ == '__main__':
    _add_b = False
    save_args(_add_b)
    main(_add_b)
