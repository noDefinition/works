from itertools import product

from sklearn.cluster import KMeans

from uclu.data.datasets import *
from uclu.data.document import Document
from utils import mu, ru, tmu
from utils.tune.tune_utils import auto_gpu


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


# def get_args_file(d_class: Data, add_body: bool):
#     postfix = 'add_body' if add_body else 'no_body'
#     args_file = './kmeans_args_{}_{}.pkl'.format(d_class.name, postfix)
#     return args_file


# def run_kmeans(log_file, features, topics, kmeans_kwarg, extra_kwarg):
#     print(kmeans_kwarg, extra_kwarg)


# def main(log_file: str, d_class: Data, add_body: bool):
# args_file = get_args_file(d_class, add_body)
# args = iu.load_pickle(args_file)
# print('len(args)', len(args), [len(k) for k in args])
# args = [(log_file, features, topics, kmeans_kwarg, extra_kwarg)
#         for features, topics, extra_kwarg in args]
# mu.multi_process(run_kmeans, args)


def run_kmeans(log_file: str, arg_file: str, kmeans_kwarg: dict, **_):
    features, topics, extra_kwarg = iu.load_pickle(arg_file)
    x_new = KMeans(**kmeans_kwarg, n_jobs=12).fit_transform(features)
    clusters = np.argmax(x_new, axis=1)
    score_kwarg = au.scores(topics, clusters)
    ret = dict(**kmeans_kwarg, **extra_kwarg, **score_kwarg)
    iu.dump_array(log_file, [ret], mode='a')
    print('---- over ---->', ret)
    # ret = kmeans_kwarg.copy()
    # ret.update(kmeans_kwarg)
    # ret.update(extra_kwarg)
    # ret.update()


# def main(log_file: str, arg_path: str):
#     kmeans_kwarg = {'max_iter': 300}
#     files = iu.list_children(arg_path, full_path=True)
#     auto_gpu(run_kmeans, [(log_file, file, kmeans_kwarg) for file in files], {0: 5})


def save_args(args_path: str, d_class: Data, add_body: bool):
    smp = Sampler(d_class)
    smp.load_basic()
    smp.fit_sparse(add_body=add_body, tfidf=True, p_num=12)
    topics = [d.tag for d in smp.docarr]
    basic = {'dn': d_class.name, 'addb': int(add_body)}

    def save2file(x, y, kw):
        file = iu.join(args_path, au.entries2name(kw, postfix='.pkl'))
        iu.dump_pickle(file, (x, y, kw))
        print(file)

    # for dim in smp.d_obj.support_dims:
    for dim in [32, 64, 128]:
        smp.prepare_embedding(dim=dim, topic_ratio=1)
        mtx = get_mean_pooling_multi(smp.docarr, smp.w_embed, add_body)
        extra = {'source': 'mean', 'dim': mtx.shape[1]}
        save2file(mtx, topics, dict(**basic, **extra))
        # extra.update(basic)
        # args.append((mtx, topics, extra))
        # print(mtx.shape, extra)

    dims = [32, 64, 128, 256, 300]
    dims = [32, 64, 128]
    tf = smp.get_feature('tf')
    tf_svds = get_svd(tf, dims)
    for mtx in tf_svds:
        extra = {'source': 'tf', 'dim': mtx.shape[1]}
        save2file(mtx, topics, dict(**basic, **extra))
        # extra.update(basic)
        # args.append((mtx, topics, extra))
        # print(mtx.shape, extra)

    tfidf = smp.get_feature('tfidf')
    tfidf_svds = get_svd(tfidf, dims)
    for mtx in tfidf_svds:
        extra = {'source': 'tfidf', 'dim': mtx.shape[1]}
        save2file(mtx, topics, dict(**basic, **extra))
        # extra.update(basic)
        # args.append((mtx, topics, extra))
        # print(mtx.shape, extra)

    # args_file = get_args_file(d_class, add_body)
    # iu.dump_pickle(args_file, args)


def save_args_and_run():
    args_path = 'kmeans_args'

    iu.mkdir(args_path, rm_prev=False)
    # d_class_range = [DataSo, DataSf, DataAu]
    d_class_range = [DataZh]
    addb_range = [True, False]
    args = product(d_class_range, addb_range)
    for dcls, addb in args:
        save_args(args_path, dcls, addb)
    print('args saved')

    log_file = 'kmeans_{}.json'.format(tmu.format_date()[2:])
    iu.remove(log_file)
    kmeans_kwarg = {'max_iter': 300}
    files = iu.list_children(args_path, ctype=iu.FILE, pattern='zh', full_path=True)
    auto_gpu(run_kmeans, [(log_file, file, kmeans_kwarg) for file in files], {0: 20})


if __name__ == '__main__':
    save_args_and_run()
