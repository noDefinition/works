from sklearn.cluster import KMeans

from clu.data.datasets import *


def fit_kmeans(x, clu_num, n_init=10, max_iter=300, n_jobs=4):
    kmeans = KMeans(n_clusters=clu_num, n_jobs=n_jobs, max_iter=max_iter, n_init=n_init)
    kmeans.fit(x)
    return kmeans


def fit_kmeans_on_data(d_class, e_dim, out_file, n_iter=5):
    import utils.reduction_utils as ru
    fp = open(out_file, mode='w')
    d_object = d_class()
    print('\nclass:{}, topic num:{}'.format(d_object.name, d_object.topic_num), file=fp)
    tf, topics = d_object.get_matrix_topics(using='tf')
    tfidf, topics = d_object.get_matrix_topics(using='tfidf')
    if e_dim > 0:
        tf = ru.fit_pca(tf, e_dim)
        tfidf = ru.fit_pca(tfidf, e_dim)
    for t, x in [('tfidf', tfidf), ('tf', tf)]:
        print(t, len(x), file=fp)
        scores_list = list()
        for i in range(n_iter):
            print(i, end=' ', flush=True, file=fp)
            clusters = fit_kmeans(x, d_object.topic_num).predict(x)
            scores = [au.score(topics, clusters, s) for s in au.eval_scores]
            scores_list.append(scores)
            print(scores, file=fp)
        nmi_list, ari_list, acc_list = au.transpose(scores_list)
        print('nmi mean: {:.4f}±{:.4f}'.format(*au.mean_std(nmi_list)), file=fp)
        print('ari mean: {:.4f}±{:.4f}'.format(*au.mean_std(ari_list)), file=fp)
        print('acc mean: {:.4f}±{:.4f}'.format(*au.mean_std(acc_list)), file=fp)


def fit_kmeans_multi():
    import utils.multiprocess_utils as mu
    args_list = [(d_class, e_dim, 'PCA_{}_e{}.txt'.format(d_class.name, e_dim))
                 for d_class in class_list for e_dim in [0, 100, 200, 300]]
    print(args_list)
    mu.multi_process_batch(fit_kmeans_on_data, batch_size=4, args_list=args_list)


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
