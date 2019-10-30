from typing import List
import numpy as np
from sklearn.cluster import KMeans

from utils import iu, mu, ru, au
from uclu.data.datasets import *
from uclu.data.document import Document


def run_kmeans(log_file, features, topics, kmeans_kwarg, extra_kwarg):
    print(kmeans_kwarg, extra_kwarg)
    x_new = KMeans(**kmeans_kwarg, n_jobs=10).fit_transform(features)
    clusters = np.argmax(x_new, axis=1)
    kmeans_kwarg.update(extra_kwarg)
    kmeans_kwarg.update(au.scores(topics, clusters))
    iu.dump_array(log_file, [kmeans_kwarg], mode='a')
    print('---- over ---->', kmeans_kwarg)


def get_mean_pooling(docarr: List[Document], w_embed, add_body):
    ret = []
    w_num, w_dim = w_embed.shape
    for doc in docarr:
        title_sum = sum((w_embed[wint] for wint in doc.title), np.zeros(w_dim))
        mean_pool = title_sum / len(doc.title)
        if add_body:
            body = list(doc.flatten_body())
            body_sum = sum((w_embed[wint] for wint in body), np.zeros(w_dim))
            body_mean = body_sum / len(body)
            mean_pool += body_mean
        ret.append(mean_pool)
    return ret


def get_mean_pooling_multi(docarr: List[Document], w_embed, add_body):
    docarr_list = mu.split_multi(docarr, 20)
    args = [(docarr, w_embed, add_body) for docarr in docarr_list]
    res_list = mu.multi_process(get_mean_pooling, args)
    mean_pooling = np.concatenate(res_list, axis=0)
    return mean_pooling


def get_svd(features, dims: List[int]):
    args = [(features, dim, 100) for dim in dims]
    res_list = mu.multi_process(ru.fit_svd, args)
    for r in res_list:
        print(r.shape)
    res_list.append(features)
    return res_list


def save_args(d_class: Data, add_body: bool):
    smp = Sampler(d_class)
    smp.load_basic()
    smp.fit_sparse(add_body=add_body, tfidf=True, p_num=20)
    topics = [d.tag for d in smp.docarr]
    base_extra = {'dn': d_class.name, 'addb': int(add_body)}

    args = []
    for dim in smp.d_obj.support_dims:
        smp.prepare_embedding(dim=dim, topic_ratio=1)
        mtx = get_mean_pooling_multi(smp.docarr, smp.w_embed, add_body)
        extra = {'source': 'mean', 'dim': mtx.shape[1]}
        extra.update(base_extra)
        args.append((mtx, topics, extra))
        print(mtx.shape, extra)

    dims = [32, 64, 128, 256, 300]
    tf = smp.get_feature('tf')
    tf_svds = get_svd(tf, dims)
    for mtx in tf_svds:
        extra = {'source': 'tf', 'dim': mtx.shape[1]}
        extra.update(base_extra)
        args.append((mtx, topics, extra))
        print(mtx.shape, extra)

    tfidf = smp.get_feature('tfidf')
    tfidf_svds = get_svd(tfidf, dims)
    for mtx in tfidf_svds:
        extra = {'source': 'tfidf', 'dim': mtx.shape[1]}
        extra.update(base_extra)
        args.append((mtx, topics, extra))
        print(mtx.shape, extra)

    args_file = get_args_file(d_class, add_body)
    iu.dump_pickle(args_file, args)


def main(log_file: str, d_class: Data, add_body: bool):
    kmeans_kwarg = {'max_iter': 300}
    args_file = get_args_file(d_class, add_body)
    args = iu.load_pickle(args_file)
    print('len(args)', len(args), [len(k) for k in args])
    args = [(log_file, features, topics, kmeans_kwarg, extra_kwarg)
            for features, topics, extra_kwarg in args]
    mu.multi_process(run_kmeans, args)


def get_args_file(d_class: Data, add_body: bool):
    postfix = 'add_body' if add_body else 'no_body'
    args_file = './kmeans_args_{}_{}.pkl'.format(d_class.name, postfix)
    return args_file


def save_args_and_run():
    from itertools import product
    d_class_range = [DataSf, DataAu]
    addb_range = [True, False]
    args = product(d_class_range, addb_range)

    # for d_class, add_b in args:
    #     save_args(d_class, add_b)
    # print('args saved')

    log_file = './kmeans.json'
    iu.remove(log_file)
    for d_class, add_b in args:
        main(log_file, d_class, add_b)


if __name__ == '__main__':
    save_args_and_run()
