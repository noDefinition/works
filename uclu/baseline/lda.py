from collections import OrderedDict as Od
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.sparse import vstack
from utils import tu

from utils import iu, mu, tu
# from clu.data.datasets import *
from uclu.data.datasets import *


def run_multi_and_output(func, batch_size, args_list, result_file):
    res_list = mu.multi_process_batch(func, batch_size, args_list)
    iu.dump_array(result_file, res_list)


def run_lda(out_file, tf_matrix, topics, lda_kwargs: dict):
    print(lda_kwargs)
    ret = lda_kwargs.copy()
    x_new = LDA(**lda_kwargs, n_jobs=1).fit_transform(tf_matrix)
    clusters = np.argmax(x_new, axis=1)
    ret.update(au.scores(topics, clusters))
    iu.dump_array(out_file, [ret], mode='a')


def main():
    add_body = False
    res_file = './lda_results_{}.json'.format('add_body' if add_body else 'no_body')
    iu.remove(res_file)
    smp = Sampler(DataSo)
    smp.load(0, 0, 0, 0)
    smp.fit_sparse(add_body=add_body, tfidf=False, p_num=28)
    tf_mtx = vstack([d.tf for d in smp.docarr])
    topics = [d.tag for d in smp.docarr]

    # ratios = np.concatenate([np.arange(0.2, 1, 0.2), np.arange(1, 5.1, 0.5)])
    ratios = np.array([1])
    iter_num = 100
    rerun_num = 2
    kwargs = tu.LY({
        'max_iter': [iter_num],
        'learning_method': ['batch'],
        'random_state': [i + np.random.randint(0, 1000) for i in range(rerun_num)],
        'doc_topic_prior': [0.1],
        'topic_word_prior': [10, 1, 0.1, 0.01],
        'n_components': list(map(int, (ratios * DataSo.topic_num))),
    }).eval()
    args = [(res_file, tf_mtx, topics, kwarg) for kwarg in kwargs]
    mu.multi_process_batch(run_lda, 10, args)


if __name__ == '__main__':
    main()
