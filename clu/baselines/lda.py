from collections import OrderedDict as Od

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA

from utils import iu, mu, tu
from clu.data.datasets import *


def run_multi_and_output(func, batch_size, args_list, result_file):
    res_list = mu.multi_process_batch(func, batch_size, args_list)
    iu.dump_array(result_file, res_list)


def run_lda(kwargs: dict):
    print(kwargs)
    kwargs_copy = kwargs.copy()
    d_class: Data = name2d_class[kwargs.pop('dname')]
    tf_matrix, topics = d_class().get_matrix_topics(using='tf')
    x_new = LDA(**kwargs, n_jobs=1).fit_transform(tf_matrix)
    clusters = np.argmax(x_new, axis=1)
    kwargs_copy.pop('learning_method')
    kwargs_copy.pop('random_state')
    return kwargs_copy, au.scores(topics, clusters)
    # return lda_kwargs, topics.tolist(), clusters.tolist()


def run_lda_using_kwargs(result_file):
    rerun_num = 4
    ratios = np.concatenate([np.arange(0.2, 1, 0.2), np.arange(1, 5.1, 0.5)])
    common = tu.LY({
        'max_iter': [200, ],
        'learning_method': ['batch', ],
        'random_state': [i + 159867 for i in range(rerun_num)],
    })
    datas = tu.LY({
        'dname': [DataTrec.name],
        'doc_topic_prior': [0.1, 0.01],
        'topic_word_prior': [0.1, 0.01],
        'n_components': (ratios * DataTrec.topic_num).astype(np.int32),
    }, {
        'dname': [DataGoogle.name],
        'doc_topic_prior': [0.1, 0.01],
        'topic_word_prior': [0.1, 0.01],
        'n_components': (ratios * DataGoogle.topic_num).astype(np.int32),
    }, {
        'dname': [DataEvent.name],
        'doc_topic_prior': [0.1, 0.01],
        'topic_word_prior': [1., 0.1],
        'n_components': (ratios * DataEvent.topic_num).astype(np.int32),
    }, {
        'dname': [DataReuters.name],
        'doc_topic_prior': [0.1, 0.01],
        'topic_word_prior': [1., 0.1],
        'n_components': (ratios * DataReuters.topic_num).astype(np.int32),
    }, {
        'dname': [Data20ng.name],
        'doc_topic_prior': [1, 0.1],
        'topic_word_prior': [1., 0.1, 0.01],
        'n_components': (ratios * Data20ng.topic_num).astype(np.int32),
    })
    # nv_list = {
    #     DataTREC: [('doc_topic_prior', [0.1, 0.01]), ('topic_word_prior', [0.1, 0.01])],
    #     DataGoogle: [('doc_topic_prior', [0.1, 0.01]), ('topic_word_prior', [0.1, 0.01])],
    #     DataEvent: [('doc_topic_prior', [0.1, 0.01]), ('topic_word_prior', [1., 0.1])],
    #     # DataReuters: [('doc_topic_prior', [0.1, 0.01]), ('topic_word_prior', [0.1])],
    #     # Data20ng: [('doc_topic_prior', [1]), ('topic_word_prior', [1., 0.1, 0.01])],
    # }[d_class]
    # common = [
    #     ('max_iter', [200, ]),
    #     ('learning_method', ['batch', ]),
    #     ('random_state', [i * 1605349 for i in range(4)]),
    #     ('n_components', [d_class.topic_num * r for r in ratios]),
    # ]
    # args_list = [(d_class, g) for g in au.grid_params(nv_list + common)]
    od_list = (common * datas).eval()
    print(len(od_list))
    res_list = mu.multi_process_batch(run_lda, 20, [(od,) for od in od_list])
    print(len(res_list))
    df = pd.DataFrame()
    for idx, (kwargs, scores) in enumerate(res_list):
        for k, v in kwargs.items():
            df.loc[idx, k] = v
        for k, v in scores.items():
            df.loc[idx, k] = v
    df.to_csv(result_file)


def analyze_mean_and_stderr(result_file):
    arg_tpc_clu_list = iu.load_array(result_file)
    rows = list()
    for kwargs, topics, clusters in arg_tpc_clu_list:
        s2v = Od((s, au.score(topics, clusters, s)) for s in au.eval_scores)
        row = Od(list(kwargs.items()) + list(s2v.items()))
        rows.append(row)
    rows = sorted(rows, key=lambda item: item['nmi'], reverse=True)
    df = pd.DataFrame(data=rows)
    # print(df)
    groups = au.group_data_frame(df, column='n_components')
    nmi_list, ari_list, acc_list = list(), list(), list()
    for _, df_ in groups:
        print(result_file)
        print(df_)
        nmis = df_['nmi'].values[0:6]
        aris = df_['ari'].values[0:6]
        accs = df_['acc'].values[0:6]
        nmi_list.append(au.mean_std(nmis))
        ari_list.append(au.mean_std(aris))
        acc_list.append(au.mean_std(accs))
    print(au.transpose(nmi_list))
    print(au.transpose(ari_list))
    print(au.transpose(acc_list))


def one_run_for_word_distribution():
    d_class = DataReuters()
    clu_num = d_class.topic_num
    print(d_class.name)
    matrix, topic_list = d_class.get_matrix_topics(using='tf')
    ifd = d_class.load_ifd()

    lda_kwargs = dict([('doc_topic_prior', 0.1), ('topic_word_prior', 0.1), ('max_iter', 200),
                       ('learning_method', 'batch'), ('random_state', 3214),
                       ('n_components', clu_num)])
    lda = LDA(**lda_kwargs, n_jobs=10)
    x_new = lda.fit_transform(matrix)
    cluid_list = np.argmax(x_new, axis=1)
    print(x_new.shape)

    # c = Counter()
    # for c_weight in x_new:
    #     region = int(np.max(c_weight) * 100)
    #     c[region] = c.setdefault(region, 0) + 1
    # for i in sorted(c.keys()):
    #     print(i, round(c[i] / len(x_new), 6))
    # print('\n---\n')
    # return
    cluid2counter = au.count_occurence(cluid_list, topic_list)
    cluid_word_distrib = lda.components_
    for cluid, word_distrib in enumerate(cluid_word_distrib):
        topic_distrib = cluid2counter[cluid].most_common()
        if len(topic_distrib) == 0:
            continue
        topic = topic_distrib[0][0]
        print(
            'cluid: {}, guess topic: {}, topic distrib: {}'.format(cluid, topic, topic_distrib[:5]))
        top_word_id = np.argsort(word_distrib)[:-31:-1]
        valid_words = [ifd.id2word(wid) for wid in top_word_id]
        print(','.join(valid_words))


if __name__ == '__main__':
    import utils.timer_utils as tmu

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print('what lda_results_0707.csv')
    run_lda_using_kwargs('lda_results_0707.csv')
    exit()
    # one_run_for_word_distribution()
    # exit()
    for _d_class in [Data20ng]:
        print('Using data:', _d_class.name)
        _topic_clu_file = 'LDA_{}.txt'.format(_d_class.name)
        tmu.check_time()
        run_lda_using_kwargs(_d_class, _topic_clu_file)
        analyze_mean_and_stderr(_topic_clu_file)
        tmu.check_time()
