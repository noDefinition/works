from sklearn.cluster import KMeans
import pandas as pd

from utils import ru, mu
from clu.data.datasets import *


def fit_kmeans(x, clu_num, n_init=10, max_iter=300, n_jobs=4):
    kmeans = KMeans(n_clusters=clu_num, n_jobs=n_jobs, max_iter=max_iter, n_init=n_init)
    kmeans.fit(x)
    return kmeans


def fit_kmeans_on_data(d_class, e_dim, out_file, pidx, n_rerun=5):
    d_obj: Data = d_class()
    tf, topics = d_obj.get_matrix_topics(using='tf')
    tfidf, _ = d_obj.get_matrix_topics(using='tfidf')
    # print('<%d>' % pidx, 'tfidf over')
    if e_dim > 0:
        tf = ru.fit_pca(tf, e_dim)
        tfidf = ru.fit_pca(tfidf, e_dim)
        # print('<%d>' % pidx, 'pca over')
    df = pd.DataFrame()
    idx = 0
    for t, x in [('tfidf', tfidf), ('tf', tf)]:
        print('<%d>' % pidx, t, x.shape)
        for t_prop in np.concatenate([np.arange(0.2, 1, 0.2), np.arange(1, 5.1, 0.5)]):
            topic_num = int(d_obj.topic_num * t_prop)
            print('<%d>' % pidx, 'topic_num', topic_num, 'data name', d_obj.name)
            df.loc[idx, 'd_name'] = d_obj.name
            df.loc[idx, 'topic_num'] = topic_num
            df.loc[idx, 'type'] = t
            scores_list = list()
            for i in range(n_rerun):
                print('<%d - %d>' % (pidx, i), end='  ', flush=True)
                clusters = fit_kmeans(x, topic_num).predict(x)
                scores = [au.score(topics, clusters, s) for s in au.eval_scores]
                scores_list.append(scores)
                # print(scores)
            score_matrix = np.array(scores_list)
            # print(score_matrix.shape)
            for s_name, s_value in zip(au.eval_scores, score_matrix.T):
                s_mean = np.mean(s_value)
                df.loc[idx, s_name] = s_mean
            idx += 1
        # nmi_list, ari_list, acc_list = au.transpose(scores_list)
        # print('nmi mean: {:.4f}±{:.4f}'.format(*au.mean_std(nmi_list)), file=fp)
        # print('ari mean: {:.4f}±{:.4f}'.format(*au.mean_std(ari_list)), file=fp)
        # print('acc mean: {:.4f}±{:.4f}'.format(*au.mean_std(acc_list)), file=fp)
    df.to_csv(out_file)
    print(' ' * 8, pidx, 'overrrrrrrrrrrrrrrrrrrrrr')


def fit_kmeans_multi():
    args_list = [
        [d_class, e_dim, './kmeans_results/PCA_{}_e{}.csv'.format(d_class.name, e_dim)]
        for d_class in [DataTREC, DataGoogle, DataEvent]
        for e_dim in [0, 100, 200, 300]
    ]
    for idx, arg in enumerate(args_list):
        arg.append(idx)
    # print(args_list)
    mu.multi_process(fit_kmeans_on_data, args_list=args_list)


# def fit_kmeans_multi():
#     import utils.reduction_utils as ru
#
#     def f(a):
#         return [round(q, 3) for q in a]
#
#     n_iter = 4
#     embed_dim = 100
#     print('embed dimension:{}'.format(embed_dim))
#     d_classes = class_list
#     for idx, d_class in enumerate(d_classes):
#         d_object = d_class()
#         print('\nclass:{}, topic num:{}'.format(d_object.name, d_object.topic_num))
#         tf, topics = d_object.get_matrix_topics(using='tf')
#         tfidf, topics = d_object.get_matrix_topics(using='tfidf')
#         # tfidf = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(tf)
#         # tfidf = np.asarray(tfidf.todense()) * np.sqrt(tfidf.shape[1])
#         # tfidf = preprocessing.normalize(tfidf, norm='l2') * 200
#         # tfidf = tfidf.astype('float32')
#         # print('tf & tfidf over')
#         tf_pca_100 = ru.fit_pca(tf, 100)
#         tfidf_pca_100 = ru.fit_pca(tfidf, 100)
#         tf_pca_200 = ru.fit_pca(tf, 200)
#         tfidf_pca_200 = ru.fit_pca(tfidf, 200)
#         print('pca over')
#         # tf_svd = ru.fit_svd(tf, embed_dim, 200)
#         # tfidf_svd = ru.fit_svd(tfidf, embed_dim, 200)
#         # print('svd over')
#
#         xs = [('tfidf_pca', tfidf_pca), ('tf_pca', tf_pca)]
#         for t, x in xs:
#             print(t)
#             scores_list = list()
#             for i in range(n_iter):
#                 print(i, end=' ', flush=True)
#                 clusters = fit_kmeans(x, d_object.topic_num).predict(x)
#                 scores = [au.score(topics, clusters, s) for s in ['nmi', 'ari']]
#                 scores_list.append(scores)
#                 print(f(scores))
#             nmi_list, ari_list = au.transpose(scores_list)
#             print('nmi mean: {}±{}'.format(*f(au.mean_std(nmi_list))))
#             print('ari mean: {}±{}'.format(*f(au.mean_std(ari_list))))
#             print()


if __name__ == '__main__':
    fit_kmeans_multi()
