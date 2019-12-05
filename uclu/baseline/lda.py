from sklearn.decomposition import LatentDirichletAllocation as LDA

from uclu.data.datasets import *
from utils import iu, mu, tu
from utils import tmu
from itertools import product


def run_lda(out_file, tf_matrix, topics, lda_kwargs: dict, extra_kwargs: dict, **_):
    print(lda_kwargs)
    x_new = LDA(**lda_kwargs, n_jobs=10).fit_transform(tf_matrix)
    clusters = np.argmax(x_new, axis=1)
    ret = lda_kwargs.copy()
    ret.update(extra_kwargs)
    ret.update(au.scores(topics, clusters))
    iu.dump_array(out_file, [ret], mode='a')


def main():
    out_file = 'lda_{}.json'.format(tmu.format_date()[2:])
    iu.remove(out_file)
    dcls_range = [DataZh]
    addb_range = [True, False]
    for dcls, addb in product(dcls_range, addb_range):
        smp = Sampler(dcls)
        smp.load_basic()
        # smp.fit_sparse(add_body=addb, tfidf=False, p_num=20)
        smp.fit_tf(add_body=addb, p_num=20)
        tf_mtx = vstack([d.tf for d in smp.docarr])
        print('tf_mtx.shape', tf_mtx.shape)
        topics = [d.tag for d in smp.docarr]

        topic_nums = list(map(int, np.array([1]) * dcls.topic_num))
        iter_num = 100
        rerun_num = 1
        kwargs_list = tu.LY({
            'max_iter': [iter_num],
            'learning_method': ['batch'],
            'random_state': [i + np.random.randint(0, 1000) for i in range(rerun_num)],
            # 'doc_topic_prior': [0.1, 1, 0.01],
            # 'topic_word_prior': [1, 0.1, 0.01],
            'doc_topic_prior': [0.1, 1],
            'topic_word_prior': [1, 0.1],
            'n_components': topic_nums,
        }).eval()
        extra_kwargs = {'addb': addb, 'dn': dcls.name}
        args = [(out_file, tf_mtx, topics, kwargs, extra_kwargs) for kwargs in kwargs_list]
        # mu.multi_process_batch(run_lda, 9, args)
        from utils.tune.tune_utils import auto_gpu
        auto_gpu(run_lda, args, {0: 9})


if __name__ == '__main__':
    main()
