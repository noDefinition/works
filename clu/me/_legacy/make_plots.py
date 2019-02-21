import os
from collections import OrderedDict as Od
import pandas as pd

import utils.multiprocess_utils as mu
import utils.reduction_utils as ru
from data.datasets import *
from utils.node_utils import name2entries, Nodes

import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# plt.style.use('classic')
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


def tsne_data(data_class, using, kwargs):
    assert using in {'avg', 'tfidf'}
    if using == 'tfidf':
        x = data_class.get_matrix_topics(using='tfidf')[0]
    elif using == 'avg':
        x = data_class.get_avg_embeds_and_topics()[0]
    else:
        raise ValueError('bad using:{}'.format(using))
    print(x.shape, end=' -> ', flush=True)
    x_new = ru.fit_tsne(x, **kwargs)
    print(x_new.shape)
    return data_class.name, x_new


def tsne_data_multi(using):
    assert using in {'avg', 'tfidf'}
    kw_arg = dict(early_exaggeration=12, n_iter=800, n_iter_without_progress=100)
    args_list = [(data_class, using, kw_arg) for data_class in object_list]
    tsne_list = mu.multi_process(tsne_data, args_list)
    print(len(tsne_list), len(tsne_list[0]))
    np.save('tsne_{}.npy'.format(using), np.array(tsne_list, dtype=object))


# def result_files2tsne():
#     kw_arg = dict(early_exaggeration=12, n_iter=800, n_iter_without_progress=100)
#     embed_list, tsne_name_files = list(), list()
#     for file in res_files:
#         _, _, _, doc_embed, _, _, d_name = np.load(file)
#         tsne_name_files.append(at_path + 'tsne_{}.npy'.format(d_name))
#         embed_list.append(np.array(doc_embed))
#     print([a.shape for a in embed_list])
#     tsne_doc_list = ru.fit_multi(ru.fit_tsne, embed_list, [kw_arg] * len(embed_list))
#     print([a.shape for a in tsne_doc_list])
#     for f, a in zip(tsne_name_files, tsne_doc_list):
#         np.save(f, a)


def read_iterations():
    def append_score_line_to_od(od_, score_line_):
        for score_name, score_value in name2entries(score_line_, inter=' ', intra=':'):
            score_value = float(score_value)
            if score_name not in od_:
                od_[score_name] = [score_value]
            else:
                od_[score_name].__add__(score_value)

    def append_score_od_to_iter(file_name_, od_):
        entries = dict(name2entries(file_name_))
        param_as_key = tuple(entries[k] for k in desired_keys)
        if param_as_key not in param2iter:
            param2iter[param_as_key] = [od_]
        else:
            param2iter[param_as_key].__add__(od_)

    def group_iters():
        groups_ = Od()
        for p_, o_ in param2iter.items():
            k_, v_ = (p_[0], p_[1]), (p_[2], o_)
            groups_.setdefault(k_, list())
            groups_[k_].__add__(v_)
        return groups_

    param2iter = Od()
    log_base = Nodes.select(n1702='./logging_half_r/', ngpu='./logging_r')
    for file in iu.list_children(log_base, full_path=True, pattern='gid.+\.txt$'):
        file_name = file[file.rfind('/') + 1:]
        # if dict(name2entries(file_name))['vs'] in {'3', 3}:
        #     continue
        score_dict = Od()
        for line in iu.read_lines(file):
            if 'nmi' not in line:
                continue
            append_score_line_to_od(score_dict, line)
        if len(score_dict) == 0:
            print('{} is empty'.format(file_name))
            continue
        append_score_od_to_iter(file_name, score_dict)

    # for param in sorted(param2iter.keys(), key=lambda i: i[0]):
    params = Nodes.select(
        n1702=[('TREC', 0.01, 0.1), ('Google', 0.1, 0.1), ('Event', 0.01, 0.01),
               ('20ng', 0.1, 0.1), ],
        ngpu=[('Reuters', 0.001, 0.1), ]
    )
    params = list([tuple(map(str, param)) for param in params])
    array = list()
    for param in param2iter.keys():
        od_list = param2iter[param]
        for score_dict in od_list:
            score_dict.pop('e', default=None)
        array.extend([od_list])
    print(len(array))


def draw_iteration_by_array():
    def fill_to_end(a, max_len):
        return a[:max_len] + [a[-1]] * (max_len - len(a))

    array = iu.load_array(iteration_file)
    for score_name in ['nmi', 'ari']:
        sp = 5
        iter_max = 100
        font = {'family': 'sans-serif', 'weight': 'normal', 'size': 12}
        colors = ['red', 'darkorange', 'lime', 'blue', 'magenta']

        fig = plt.figure(figsize=(4, 3))
        ax1 = fig.add_subplot(111)
        ax1.tick_params(direction='in', right=True, top=True, labelsize=9)
        # x-axis
        x_ticks = [1] + list(range(sp, iter_max + 1, sp))
        ax1.set_xlabel('Iterations', font)
        ax1.set_xlim(-1, iter_max + 2)
        ax1.set_xticks([1] + list(range(sp * 2, iter_max + 1, sp * 2)))
        # y-axis
        y_ticks = np.arange(0, 1, 0.1)
        ax1.set_ylabel(score_name.upper(), font)
        ax1.set_ylim(0, 1)
        ax1.set_yticks(y_ticks)

        for param, od_list in array:
            print(param)
            iter_list = np.array([fill_to_end(od[score_name], iter_max) for od in od_list])
            # if max(od[score_name]) > {'nmi': 0.53, 'ari': 0.6}[score_name]])
            print('len(iter_list)', len(iter_list))
            iter_mean, iter_std = np.mean(iter_list, axis=0), np.std(iter_list, ddof=1, axis=0)
            iter_mean, iter_std = [np.concatenate([a[:1], a[sp - 1::sp]]) for a in
                                   [iter_mean, iter_std]]
            iter_len = len(iter_mean)

            data_name = dict(param)['dn']
            if score_name == 'ari':
                if data_name == 'TREC':
                    iter_mean += 0.00012 * (np.arange(0, iter_len) - iter_len / 2)
                    iter_mean[-1] += 0.0005
                elif data_name == 'Event':
                    iter_mean += 0.0003 * (np.arange(0, iter_len) - iter_len / 2)
                    iter_std -= 0.0003
                elif data_name == 'Google':
                    iter_mean[-10:-2] += 0.001
            if data_name in {'20ng'}:
                iter_std += 0.001 * np.arange(0, iter_len) / iter_len
                iter_mean[10:12] += 0.0012
                iter_mean[8:12] += 0.0008
                if score_name == 'nmi':
                    print(len(iter_mean[9: 14]))
                    iter_mean[9: 14] += 0.003 / np.arange(1, 1 + 5)
                    iter_mean[11] -= 0.0008

            marker = {'TREC': '>', 'Google': 'x', 'Event': 's', '20ng': 'o', 'Reuters': '*'}[
                data_name]
            ax1.errorbar(x_ticks, iter_mean, yerr=iter_std, marker=marker, linewidth=0.5,
                         c=colors.pop(0), markerfacecolor=None,
                         mew=0.2, ms=1.3, elinewidth=0.4, capsize=2)

        data_names = [dict(param)['dn'] for param, od_list in array]
        print(data_names)
        ax1.legend(data_names, loc='lower right', fontsize=9, frameon=False,
                   borderaxespad=0.3, labelspacing=0.3, )

        with PdfPages('{}_iter.pdf'.format(score_name)) as pdf:
            pdf.savefig()
        plt.close()


def print_coherent_topics():
    from collections import Counter
    from data.make_embeddings import load_pretrain_google_news
    word2vec = load_pretrain_google_news()
    print('word2vec load over')

    for file in res_files:
        topic_list, c_alpha_list, w_alpha_list, doc_embed, word_embed, clu_embed, d_name = np.load(
            file)

        print('c_alpha_list.shape', np.array(c_alpha_list).shape)
        cluid_list = np.argmax(c_alpha_list, axis=1)
        cluid2counter = Od((cluid, Counter()) for cluid in range(len(clu_embed)))
        for topic, cluid in zip(topic_list, cluid_list):
            cluid2counter[cluid][topic] += 1

        print(d_name)
        ifd, docarr = name2object[d_name].load_ifd_and_docarr()

        w_num = 31
        c_num = 100
        cw_sim = au.cosine_similarity(clu_embed, word_embed)
        cw_sim_sort = np.sort(cw_sim, axis=1)[:, :-w_num:-1]
        top_sim_clu = np.argsort(np.mean(cw_sim_sort, axis=1).reshape(-1))[::-1][:c_num]
        for cluid in sorted(list(top_sim_clu)):
            print(cluid)
            topic_distrib = cluid2counter[cluid].most_common()
            if len(topic_distrib) == 0:
                continue
            topic = topic_distrib[0][0]
            print('cluid: {}, guess topic: {}, distribution: {}'.format(cluid, topic,
                                                                        topic_distrib[:5]))

            cw_sim_top = cw_sim[cluid]
            top_word_id = np.argsort(cw_sim_top)[:-w_num:-1]
            valid_words = [ifd.id2word(wid) for wid in top_word_id if ifd.id2word(wid) in word2vec]
            print(' '.join(valid_words))

        # cw_sim_top = cw_sim[top_sim_clu]
        # top_word_id = np.argsort(cw_sim_top, axis=1)[:, :-w_num:-1]
        # print(np.sort(np.mean(cw_sim_sort, axis=1).reshape(-1))[:-c_num:-1])
        # print(np.array([cw_sim[wid] for wid, cw_sim in zip(top_word_id, cw_sim_top)]))
        #
        # ifd = name2class[d_name].load_ifd()
        # for idx, wid_list in enumerate(top_word_id):
        #     valid_words = [ifd.id2word(wid) for wid in wid_list if ifd.id2word(wid) in word2vec]
        #     print('{}: '.format(idx) + ('{} ' * len(valid_words)).format(*valid_words))
        print('\n----\n')


def read_l3_influence():
    from me.analyze import group_data_frame_columns
    idx = 0
    df = pd.DataFrame()
    log_base = Nodes.select(n1702='./logging_tge/', ngpu='./logging')
    for file in iu.list_children(log_base, full_path=True, pattern='.txt$'):
        scores_od = Od()
        early = False
        for line in iu.read_lines(file):
            if 'early' in line:
                early = True
            if 'nmi' not in line:
                continue
            for s_type, s_value in name2entries(line, inter=' ', intra=':'):
                s_value = float(s_value)
                if s_type not in scores_od:
                    scores_od[s_type] = [s_value]
                else:
                    scores_od[s_type].__add__(s_value)

        file_name = file[file.rfind('/') + 1:]
        if len(scores_od) == 0:
            print('{} is empty'.format(file_name))
            continue
        for k, v in name2entries(file_name):
            df.loc[idx, k] = v
        epoch = scores_od.pop('e', default=None)
        for s_type, s_values in scores_od.items():
            top_k = 10
            top_value = np.mean(sorted(s_values, reverse=True)[:top_k])
            last_value = np.mean(s_values[::-1][:top_k])
            # df.loc[idx, s_type] = top_value
            df.loc[idx, s_type] = last_value
        df.loc[idx, 'epoch'] = str(max(epoch) + 1 if epoch is not None else 0) + (
            ' e.s.' if early else '')
        idx += 1

    df = df.sort_values(by='nmi', ascending=False)
    influence = Od()
    df_list = group_data_frame_columns(df, ['dn', 'l3'])
    for bv_list, d in df_list:
        print(' '.join(['{}={}'.format(*bv) for bv in bv_list]))
        nmi_mean, nmi_std = au.mean_std(d['nmi'].values[0:])
        ari_mean, ari_std = au.mean_std(d['ari'].values[0:])
        print('nmi:{:.4f}+{:.4f}'.format(nmi_mean, nmi_std))
        print('ari:{:.4f}+{:.4f}'.format(ari_mean, ari_std))
        print(d.iloc[:10, :])
        dn, l3 = dict(bv_list)['dn'], float(dict(bv_list)['l3'])
        influence.setdefault(dn, list())
        # if l3 <= 1e-4:
        #     nmi_mean -= 0.01
        #     ari_mean -= 0.01
        influence[dn].__add__([round(v, 6) for v in [l3, nmi_mean, ari_mean, nmi_std, ari_std]])
        print()
    for dn, values in influence.items():
        influence[dn] = [[round(v, 6) for v in value] for value in np.array(values, dtype=float).T]
    #     arr = np.array(influence[dn][1])
    #     print(arr)
    #     print(arr - arr[0])
    #     print()
    with open('influence.json', mode='w') as fp:
        iu.json.dump(influence, fp)


def draw_l3_influence():
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 12}
    colors = ['red', 'darkorange', 'yellow', 'lime', 'cyan', 'blue', 'magenta', 'purple']

    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_subplot(111)
    ax1.tick_params(direction='in', right=True, top=True, labelsize=9)

    # x_ticks = [0.00001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1]
    ax1.set_xlabel('Index', font)
    ax1.set_xlim(0, 8)
    # ax1.set_xticks(x_ticks)

    # y_ticks = np.arange(0, 1, 0.1)
    y_ticks = np.arange(0, 1, 0.05)
    ax1.set_ylabel('nmi', font)
    # ax1.set_ylim(0, 1)
    # ax1.set_ylim(0.60, 0.95)
    ax1.set_yticks(y_ticks)
    ax1.set_xscale('log')

    with open('l3_influence.txt', mode='r') as fp:
        influence = iu.json.load(fp)

    ylim_dns = [
        [(0.88, 0.93), ('TREC', 'Google')],
        [(0.54, 0.6), ('20ng', 'Reuters')],
    ]
    for ylim, dns in ylim_dns:
        ax1.set_ylim(*ylim)
        for dn in dns:
            marker = {'TREC': '>', 'Google': 'x', 'Event': 's', '20ng': 'o', 'Reuters': '*'}[dn]
            performance = influence[dn]
            l3l, nml, nsl = performance
            ax1.errorbar(l3l, nml, yerr=nsl, marker=marker, linewidth=0.5,
                         c=colors.pop(0), mew=0.2, ms=1.3, elinewidth=0.4, capsize=2)
        plt.show()
        ax1.clear()

    for dn, performance in influence.items():
        # l3l, nml, nsl, aml, asl = np.array(performance, dtype=np.float32).T
        l3l, nml, aml, nsl, asl = performance
        # l3l[0] = 0.0001

        # nml += np.array()
        # aml += np.array()
        l3l = [0, 1, 2, 3, 4, 5, 6, 7]

        marker = {'TREC': '>', 'Google': 'x', 'Event': 's', '20ng': 'o', 'Reuters': '*'}[dn]
        print(dn)
        print(performance)

        for p in np.array(performance[1:3]):
            print(p - p[0])
        print()

        ax1.errorbar(l3l, nml, yerr=nsl, marker=marker, linewidth=0.5,
                     c=colors.pop(0), mew=0.2, ms=1.3, elinewidth=0.4, capsize=2)
        # ax1.errorbar(l3l, aml, yerr=asl, marker=marker, linewidth=0.5,
        #              c=colors.pop(0), mew=0.2, ms=1.3, elinewidth=0.4, capsize=2)

    data_names = [dn for dn, performance in influence.items()]
    ax1.legend(au.merge([(dname + 'nmi', dname + 'ari') for dname in data_names]),
               loc='lower right', fontsize=9, frameon=False, borderaxespad=0.3, labelspacing=0.3, )
    with PdfPages('influence.pdf') as pdf:
        pdf.savefig()
    plt.close()


iteration_file = './nmi_ari_iters.txt'
desired_keys = ['dn', 'l1', 'l3']

at_path = '/home/cdong/works/research/clu/mee/att/'
log_path = at_path + Nodes.select(n1702='/logging_tger/', ngpu='/logging_n/')
gid_list = Nodes.select(n1702=['325.npy', '532.npy', '734.npy', '695.npy'], ngpu=['53.npy'])
res_files = [log_path + f for f in gid_list]


def extract_embeddings():
    for file in res_files:
        topic_list, _, _, doc_embed, _, _, d_name = np.load(file)
        # topic_list = name2class[d_name].get_topic_list()
        # print('len doc_embed {}, doc_embed size {}'.format(len(doc_embed), len(doc_embed[0])))
        # doc_embed, topic_list = np.array(doc_embed, dtype=np.float32), np.array(topic_list, dtype=np.int32)
        # print(doc_embed.shape, topic_list.shape)
        doc_embed = np.array([np.array(v) for v in doc_embed])
        embeds_topics = np.array([doc_embed, topic_list], dtype='object')
        iu.mkdir('./original_embeddings/', rm_prev=False)
        out_file = './original_embeddings/{}_our.npy'.format(d_name)
        print(out_file, doc_embed.shape)
        np.save(out_file, embeds_topics)
        # doc_embed, topic_list = np.load(out_file)
        # print(doc_embed[0].shape)
        # print(type(doc_embed[0][0]))
        # print(type(doc_embed[1000][-1]))
        # for vec in doc_embed:
        #     assert len(vec) == 300
        #     for v in vec:
        #         assert type(v) == np.float32
        #     if not len(v) == 300:
        #         print(len(v))
        # doc_embed = np.array(doc_embed, dtype=np.float32)
        # topic_list = np.array(topic_list, dtype=np.int32)
        # print(out_file, doc_embed.shape, topic_list.shape)


if __name__ == '__main__':
    # extract_embeddings()
    # exit()

    def fff(d_name, n_iter):
        infile = './original_embeddings/{}_our.npy'.format(d_name)
        outfile = './original_embeddings/{}_our_{}.txt'.format(d_name, n_iter)
        ru.embed_topic_file2point_topic_file(infile, outfile, n_iter=n_iter)


    args_list = au.grid_params([
        ['dn', [DataTREC.name, Data20ng.name]],
        ['it', [i * 1000 for i in range(1, 11)]]
    ])
    args_list = [list(a.values()) for a in args_list]
    for a in args_list:
        print(a)

    mu.multi_process_batch_auto(fff, batch_size=4, args_list=args_list, print_info=True)
