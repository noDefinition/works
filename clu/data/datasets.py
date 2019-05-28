from collections import Counter
from typing import List

import numpy as np
from utils import au, du, iu, pu, mu
from utils.doc_utils import Document


class Data:
    stem = True
    name = orgn = topic_num = None
    wf_flt_func = doc_flt_func = topic_flt_func = None

    def __init__(self):
        # self.orgn_file, = self.fill(self.orgn)
        # self.temp_file, self.docarr_file, self.dict_file = \
        #     self.fill([self.name + s for s in ['_temp.txt', '_docarr.txt', '_dict.txt']])
        # self.embed_init_file, self.word2vec_file = \
        # fill([self.name + s for s in ['_clu_init_300.npy', '_word2vec_300.npy']])
        self.support_dims = [64, 128, 300]
        self.support_ratios = list(np.arange(1, 6) / 5) + list(np.arange(2, 6))
        self.word2wid_file = self.fill_new(self.name + '_word2wid.txt')
        self.new_docarr_file = self.fill_new(self.name + '_docarr.txt')
        self.docarr = self.word2wid = None

    def __call__(self, *args, **kwargs):
        return self

    # def fill(self, files):
    #     base = '/home/cdong/works/clu/data/input_and_outputs/short_text_corpus'
    #     if isinstance(files, list):
    #         return [iu.join(base, self.name, f) for f in files]
    #     elif isinstance(files, str):
    #         return iu.join(base, self.name, files)
    #     else:
    #         raise ValueError('wtf')

    def fill_new(self, files):
        base = '/home/cdong/works/clu/data/corpus'
        if isinstance(files, list):
            return [iu.join(base, self.name, f) for f in files]
        elif isinstance(files, str):
            return iu.join(base, self.name, files)
        else:
            raise ValueError('wtf')

    def get_word2vec_file(self, embed_dim: int, required=True):
        assert embed_dim in self.support_dims
        word2vec_file = self.fill_new(self.name + '_word2vec_{}.pkl'.format(embed_dim))
        if required and not iu.exists(word2vec_file):
            raise ValueError('path %s does not exists' % word2vec_file)
        return word2vec_file

    def get_clu_init_file(self, embed_dim: int, topic_ratio: float, required=True):
        assert embed_dim in self.support_dims
        clu_init_file = self.fill_new(
            self.name + '_clu_init_{}_{:.2f}.pkl'.format(embed_dim, topic_ratio))
        if required and not iu.exists(clu_init_file):
            raise ValueError('path %s does not exists' % clu_init_file)
        return clu_init_file

    # def load_ifd_and_docarr(self):
    #     return self.load_ifd(), self.load_docarr()
    #
    # def load_docarr(self) -> t.List:
    #     if self.docarr is None:
    #         self.docarr = du.load_docarr(self.docarr_file)
    #     return self.docarr
    #
    # def load_ifd(self):
    #     from utils.id_freq_dict import IdFreqDict
    #     return IdFreqDict().load_dict(self.dict_file)
    #
    # def load_word2vec(self):
    #     wv_list = np.load(self.word2vec_file)
    #     print('len(wv_array) {}, len(wv_array[0]) {}, embed dim {}'.format(
    #         len(wv_list), len(wv_list[0]), len(wv_list[0][1])))
    #     return dict(wv_list)

    def load_new_docarr(self) -> List[Document]:
        if self.docarr is None:
            self.docarr = du.load_docarr(self.new_docarr_file)
        return self.docarr

    def load_word2wid(self) -> dict:
        if self.word2wid is None:
            self.word2wid = iu.load_json(self.word2wid_file)
        return self.word2wid

    def load_word2vec(self, embed_dim: int) -> dict:
        word2vec_file = self.get_word2vec_file(embed_dim)
        return iu.load_pickle(word2vec_file)

    def load_clu_init(self, embed_dim: int, topic_ratio: float = 1):
        clu_init_file = self.get_clu_init_file(embed_dim, topic_ratio)
        return iu.load_pickle(clu_init_file)

    def train_word2vec(self, embed_dim: int):
        from utils.node_utils import Nodes
        from gensim.models.word2vec import Word2Vec
        docarr = self.load_new_docarr()
        word2wid = self.load_word2wid()
        word2vec_file = self.get_word2vec_file(embed_dim, required=False)
        tokens_list = [d.tokens for d in docarr]
        print('start training word2vec')
        model = Word2Vec(min_count=0, workers=Nodes.max_cpu_num(), iter=300, size=embed_dim)
        model.build_vocab(tokens_list)
        model.train(tokens_list, total_examples=model.corpus_count, epochs=300)
        word2vec = {w: model[w] for w in word2wid}
        print('word2vec over, vocab size:', len(word2vec))
        iu.dump_pickle(word2vec_file, word2vec)

    def train_centroids(self, embed_dim: int, topic_ratio: float):
        from clu.baselines.kmeans import fit_kmeans
        docarr = self.load_new_docarr()
        word2vec = self.load_word2vec(embed_dim)
        clu_init_file = self.get_clu_init_file(embed_dim, topic_ratio, required=False)
        avg_embeds = list()
        for doc in docarr:
            avg = np.mean([word2vec[word] for word in doc.tokens], axis=0)
            avg_embeds.append(avg)
        # ifd, word2vec = self.load_ifd(), self.load_word2vec()
        # wid2vec = [None] * len(word2wid)
        # for word, wid in word2wid.items():
        #     wid2vec[wid - 1] = word2vec[word]
        #     # for w in ifd.vocabulary():
        #     # wid2vec[ifd.word2id(w)] = word2vec[w]
        # for v in wid2vec:
        #     assert v is not None
        # wid2vec = np.array(wid2vec, dtype=np.float32)
        # avg_embeds, _ = self.get_avg_embeds_and_topics()
        centroids = fit_kmeans(avg_embeds, int(self.topic_num * topic_ratio)).cluster_centers_
        print('centroids.shape:{}, to file:{}'.format(centroids.shape, clu_init_file))
        iu.dump_pickle(clu_init_file, centroids)

    def get_topics(self):
        return [d.topic for d in self.load_docarr()]

    def get_matrix_topics(self, using):
        assert using in {'tf', 'tfidf'}
        ifd, docarr = self.load_ifd_and_docarr()
        matrix = None
        if using == 'tf':
            matrix = [d.tf for d in du.docarr_fit_tf(ifd, docarr)]
        if using == 'tfidf':
            matrix = [d.tfidf for d in du.docarr_fit_tfidf(ifd, docarr)]
        topics = [d.topic for d in docarr]
        return np.array(matrix), np.array(topics)

    def get_matrix_topics_for_dec(self):
        from sklearn.feature_extraction.text import TfidfTransformer
        matrix, topics = self.get_matrix_topics(using='tf')
        topics = np.array(au.reindex(topics))
        matrix = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(matrix)
        matrix = matrix.astype(np.float32)
        print(matrix.shape, matrix.dtype, matrix.size)
        matrix = np.asarray(matrix.todense()) * np.sqrt(matrix.shape[1])
        print('todense succeed')
        p = np.random.permutation(matrix.shape[0])
        matrix = matrix[p]
        topics = topics[p]
        print('permutation finished')
        assert matrix.shape[0] == topics.shape[0]
        return matrix, topics

    def get_matrix_topics_for_vade(self):
        ifd, docarr = self.load_ifd_and_docarr()
        topics = [d.topic for d in docarr]
        matrix = [d.tfidf for d in du.docarr_fit_tfidf(ifd, docarr)]
        return np.array(matrix), np.array(topics)

    def get_avg_embeds_and_topics(self, embed_dim):
        docarr = self.load_new_docarr()
        word2vec = self.load_word2vec(embed_dim)
        e_dim = len(list(word2vec.values())[0])
        vs_list = [[word2vec[w] for w in d.tokens if w in word2vec] for d in docarr]
        avg_embeds = [np.zeros(e_dim) if len(vs) == 0 else np.mean(vs, axis=0) for vs in vs_list]
        print('get avg: 0-len doc: {}/{}'.format(
            sum(1 for v in vs_list if len(v) > 0), len(docarr)))
        return np.array(avg_embeds), np.array([d.topic for d in docarr])

    def filter_from_temp(self):
        c = self.__class__
        topic_flt_func, wf_flt_func, doc_flt_func = c.topic_flt_func, c.wf_flt_func, c.doc_flt_func
        docarr = du.load_docarr(self.temp_file)
        docarr = du.filter_duplicate_docid(docarr)
        docarr = du.tokenize_docarr(docarr, stemming=self.stem)
        print('data prepare (filter duplicate id, tokenize) over')
        acc_topics, rej_topics_1, docarr = du.filter_docarr_by_topic(docarr, topic_flt_func)
        docarr, ifd = du.docarr_bootstrap_ifd(docarr, wf_flt_func)
        # docarr = du.filter_docarr(docarr, doc_flt_func)
        # docarr, ifd = du.docarr_bootstrap_ifd(docarr, wf_flt_func)
        # acc_topics, rej_topics_2, docarr = du.filter_docarr_by_topic(docarr, topic_flt_func)
        # docarr, ifd = du.docarr_bootstrap_ifd(docarr, wf_flt_func)
        # docarr = du.filter_docarr(docarr, doc_flt_func)

        # rej_topics = rej_topics_1 + rej_topics_2
        rej_topics = rej_topics_1
        print('get {} docs\n'.format(len(docarr)))
        print('{} suff topic:{}\n'.format(len(acc_topics), acc_topics.most_common()))
        print('{} insuff topic:{}'.format(len(rej_topics), rej_topics.most_common()[:20]))
        docarr = sorted(docarr, key=lambda d: d.topic)
        ifd.dump_dict(self.dict_file)
        du.dump_docarr(self.docarr_file, docarr)

    def filter_into_temp(self):
        raise RuntimeError('This function should be implemented in sub classes')


class DataTREC(Data):
    name = 'TREC'
    orgn = ['Tweets.txt']
    topic_num = 128
    w_verify_func = du.word_verify(3, 14, 0.8, pu.my_stop_words)
    wf_flt_func = lambda word, freq: freq >= 3
    doc_flt_func = lambda d: len(d.tokens) >= 5 and d.topic is not None
    topic_flt_func = lambda rank, freq: 10 <= freq

    def filter_into_temp(self):
        twarr = iu.load_array(self.orgn_file)
        outrows = list()
        for idx, tw in enumerate(twarr):
            if tw['relevance'] > 1:
                continue
            docid, topic, text = tw['tweetId'], tw['clusterNo'], tw['text']
            if not 10 < len(' '.join(pu.tokenize(text, pu.tokenize_pattern))):
                continue
            outrows.append([docid, topic, text])
        topics = Counter([r[1] for r in outrows])
        print('get {} rows'.format(len(outrows)))
        print('{} topics, {}'.format(len(topics), topics))
        du.dump_docarr(self.temp_file, du.make_docarr(outrows))


class DataGoogle(Data):
    name = 'Google'
    orgn = ['News.txt']
    topic_num = 152
    w_verify_func = du.word_verify(None, None, 0.0, None)
    wf_flt_func = lambda word, freq: freq >= 0
    doc_flt_func = lambda d: len(d.tokens) >= 3 and d.topic is not None
    topic_flt_func = lambda rank, freq: True

    def filter_into_temp(self):
        twarr = iu.load_array(self.orgn_file)
        print(len(twarr), type(twarr[0]))
        docarr = du.make_docarr(
            [[tw[k] for k in ('tweetId', 'clusterNo', 'textCleaned')] for tw in twarr])
        du.dump_docarr(self.temp_file, docarr)


class DataEvent(Data):
    name = 'Event'
    orgn = ['Terrorist']
    topic_num = 69
    w_verify_func = du.word_verify(2, 16, 0.8, pu.nltk_stop_words)
    wf_flt_func = lambda word, freq: freq >= 3
    doc_flt_func = lambda d: len(d.tokens) >= 3 and d.topic is not None
    topic_flt_func = lambda rank, freq: True

    def filter_into_temp(self):
        file_list = iu.list_children(self.orgn_file, full_path=True)
        twarr_list = [iu.load_array(file) for file in file_list]
        doclist = list()
        for topic_id, twarr in enumerate(twarr_list):
            for tw in twarr:
                doclist.append((str(tw['id']), topic_id, tw['text'].replace('#', '')))
        docarr = du.make_docarr(doclist)
        du.dump_docarr(self.temp_file, docarr)


class Data20ng(Data):
    name = '20ng'
    orgn = ['20ng']
    topic_num = 20
    wf_flt_func = lambda word, freq: freq >= 8 and 3 <= len(word) <= 14 and pu.is_valid_word(word)
    doc_flt_func = lambda d: len(d.tokens) >= 4
    topic_flt_func = lambda rank, freq: True

    def filter_into_temp(self):
        json_list = iu.load_array(self.orgn_file)
        item_list = list()
        for i, o in enumerate(json_list):
            text = ' '.join(pu.tokenize(o['text'], pu.tokenize_pattern)[:1200])
            # text = ' '.join(pu.tokenize(o['text'], pu.tokenize_pattern)[:3000])
            # text = o['text']
            item_list.append((i, o['cluster'], text))
        docarr = du.make_docarr(item_list)
        du.dump_docarr(self.temp_file, docarr)


class DataReuters(Data):
    name = 'Reuters'
    orgn = ['segments']
    topic_num = 31
    w_verify_func = du.word_verify(3, 16, 0.8, pu.nltk_stop_words)
    wf_flt_func = lambda word, freq: freq >= 3
    doc_flt_func = lambda d: len(d.tokens) >= 3 and d.topic is not None
    topic_flt_func = lambda rank, freq: freq >= 20

    def filter_into_temp(self):
        from bs4 import BeautifulSoup
        files = iu.list_children(self.orgn_file, full_path=True)
        array = list()
        for fidx, file in enumerate(files):
            print(fidx)
            tree = BeautifulSoup(''.join(iu.read_lines(file)), "html.parser")
            for article in tree.find_all("reuters"):
                topics = list(article.topics.children)
                if not len(topics) == 1:
                    continue
                topic = str(topics[0].text.encode('ascii', 'ignore'))
                text = article.find('text')
                if text is None or text.body is None:
                    continue
                title = str(
                    text.title.text.encode('utf-8', 'ignore')) if text.title is not None else ''
                title = ' '.join(pu.tokenize(title, pu.tokenize_pattern))
                body = str(text.body.text.encode('utf-8', 'ignore'))
                body = ' '.join(pu.tokenize(body, pu.tokenize_pattern))
                array.append((topic, '{}, {}'.format(title, body)))
        docarr = du.make_docarr([(idx, topic, body) for idx, (topic, body) in enumerate(array)])
        print(len(docarr))
        print(Counter([d.topic for d in docarr]))
        print(len(sorted(set([d.topic for d in docarr]))))
        du.dump_docarr(self.temp_file, docarr)


class_list = [DataTREC, DataGoogle, DataEvent, DataReuters, Data20ng]
object_list = [_c() for _c in class_list]
name_list = [_c.name for _c in class_list]
name2d_class = dict(zip(name_list, class_list))
name2object = dict(zip(name_list, object_list))
dft = object()


# def data_select(c, t=dft, g=dft, e=dft, r=dft, n=dft, r10=dft, default=dft):
#     lookup = {
#         DataTREC: t, DataTREC.name: t,
#         DataGoogle: g, DataGoogle.name: g,
#         DataEvent: e, DataEvent.name: e,
#         DataReuters: r, DataReuters.name: r,
#         Data20ng: n, Data20ng.name: n,
#         # DataR10K: r10, DataR10K.name: r10,
#     }
#     key = c if c in lookup else c.__class__ if c.__class__ in lookup else None
#     value = lookup[key]
#     if value is dft:
#         if default is dft:
#             raise ValueError('default value not given')
#         value = default
#     return value

class Sampler:
    def __init__(self, d_cls):
        if d_cls in name2d_class:
            d_cls = name2d_class[d_cls]
        self.d_obj: Data = d_cls()
        self.docarr: List[Document] = None
        self.word2wid: dict = None
        self.word2vec: dict = None
        self.clu_embed_init = self.word_embed_init = None

    def load(self, embed_dim: int = 64, topic_ratio: float = 1):
        self.docarr = self.d_obj.load_new_docarr()
        self.word2wid = self.d_obj.load_word2wid()
        self.word2vec = self.d_obj.load_word2vec(embed_dim)
        self.clu_embed_init = self.d_obj.load_clu_init(embed_dim, topic_ratio)
        self.word_embed_init = [None] * len(self.word2vec)
        for word, wid in self.word2wid.items():
            self.word_embed_init[wid - 1] = self.word2vec[word]
        self.word_embed_init = np.array(self.word_embed_init)
        for e in self.word_embed_init:
            assert e is not None
        # max_wid, min_wid = -1, 1e9
        # for d in self.docarr:
        #     max_wid = max(max_wid, max(d.tokenids))
        #     min_wid = min(min_wid, min(d.tokenids))
        # print('wid in docs:', max_wid, min_wid)
        # wids = self.word2wid.values()
        # print('wid in word2wid:', max(wids), min(wids))
        # self.ifd, self.docarr = self.d_obj.load_ifd_and_docarr()
        # assert self.v_size == max(max(d.tokenids) for d in self.docarr) + 1
        self.eval_batches = self.split_length(self.docarr, 256)

    # def fit_tfidf(self):
    #     du.docarr_fit_tfidf(self.ifd, self.docarr)

    @staticmethod
    def split_length(docarr: List[Document], batch_size: int) -> List[List[Document]]:
        docarr = sorted(docarr, key=lambda x: len(x.tokenids))
        batches, batch = list(), list()
        prev_len = len(docarr[0].tokenids)
        for i, d in enumerate(docarr):
            if len(d.tokenids) != prev_len or len(batch) >= batch_size:
                batches.append(batch)
                batch = [d]
            else:
                batch.append(d)
            if i >= len(docarr) - 1:
                batches.append(batch)
            prev_len = len(d.tokenids)
        return batches

    def shuffle_generate(self, batch_size, neg_batch_num):
        docarr = au.shuffle(self.docarr)
        batches = self.split_length(docarr, batch_size)
        print('shuffle_generate - batch num:', len(batches))
        i_range = range(len(batches))
        for i in i_range:
            p_batch: List[Document] = batches[i]
            n_idxes: List[int] = np.random.choice([j for j in i_range if j != i], neg_batch_num)
            n_batches: List[List[Document]] = [batches[j] for j in n_idxes]
            yield p_batch, n_batches

    # def trick_generate(self, batch_size, neg_batch_num):
    #     def fill_array(array, pos):
    #         max_seq_len = min(max(len(d.tokenids) for d in array), 2000)
    #         for d in array:
    #             tokens_fill = d.tokenids + [self.v_size] * (max_seq_len - len(d.tokenids))
    #             if len(tokens_fill) > max_seq_len:
    #                 tokens_fill = tokens_fill[:max_seq_len]
    #             # d.neg_fill = d.pos_fill = tokens_fill[:max_seq_len]
    #             if pos:
    #                 d.pos_fill = tokens_fill
    #             else:
    #                 d.neg_fill = tokens_fill
    #
    #     # docarr = sorted(self.docarr, key=lambda d: len(d.tokenids))
    #     i = 0
    #     self.eval_batches = list()
    #     docarr = au.shuffle(self.docarr)
    #     doc_num = len(docarr)
    #     for start, until in au.split_since_until(doc_num, batch_size):
    #         n_range = list(range(0, start)) + list(range(until, doc_num))
    #         n_idxes = np.random.choice(n_range, size=batch_size * neg_batch_num)
    #         p_batch = docarr[start: until]
    #         n_batch = [docarr[i] for i in n_idxes]
    #         fill_array(p_batch, pos=True)
    #         fill_array(n_batch, pos=False)
    #         self.eval_batches.append(p_batch)
    #         yield i, p_batch, n_batch
    #         i += 1
    #
    # def trick_generate_many(self, batch_size, neg_batch_num):
    #     print('trick_generate_many')
    #     from collections import OrderedDict as Od
    #     j = 0
    #     topic_od = Od()
    #     for d in self.docarr:
    #         topic_od.setdefault(d.topic, list()).__add__(d)
    #     for topic, topic_docarr in topic_od.items():
    #         batches = self.split_length(topic_docarr, batch_size)
    #         i_range = range(len(batches))
    #         for i in i_range:
    #             p_batch = batches[i]
    #             n_idxes = np.random.choice([j for j in i_range if j != i], neg_batch_num)
    #             n_batches = [batches[j] for j in n_idxes]
    #             yield j, p_batch, n_batches
    #             j += 1


# def summary_datasets():
#     import pandas as pd
#     df = pd.DataFrame(columns=['K', 'D', 'V', 'Avg-len', 'Max-len', 'Min-len'])
#     for idx, cls in enumerate([DataTREC, DataGoogle, DataEvent, Data20ng, DataReuters]):
#         obj = cls()
#         print(obj.docarr_file)
#         ifd, docarr = obj.load_ifd_and_docarr()
#         len_arr = [len(d.tokenids) for d in docarr]
#         K = len(set([d.topic for d in docarr]))
#         D = len(docarr)
#         V = ifd.vocabulary_size()
#         Avg_len = round(float(np.mean(len_arr)), 2)
#         Max_len = round(float(np.max(len_arr)), 2)
#         Min_len = round(float(np.min(len_arr)), 2)
#         df.loc[obj.name] = [K, D, V, Avg_len, Max_len, Min_len]
#     print(df)
#     df.to_csv('summary.csv')
#
#
# def to_btm():
#     base = '/home/cdong/works/research/clu/data/'
#     for _o in object_list:
#         docarr = _o.load_docarr()
#         lines = ['{}\t{}|{}'.format(i, d.topic, ' '.join(map(str, d.tokenids)))
#                  for i, d in enumerate(docarr)]
#         iu.write_lines(base + '{}_btm.txt'.format(_o.name), lines)

def transfer_files(d_class: Data):
    data = d_class()
    print(data.name)
    print(data.new_docarr_file)
    print(data.word2wid_file)
    print()
    for dim in data.support_dims:
        for ratio in [1]:
            print(data.name, dim, ratio)
            data.train_centroids(dim, ratio)
    # data_path = iu.parent_name(data.new_docarr_file)
    # iu.mkprnts(data.new_docarr_file)
    #
    # doc_arr = du.load_docarr(data.docarr_file)
    # for doc in doc_arr:
    #     doc.tokenids = [i + 1 for i in doc.tokenids]
    # du.dump_docarr(data.new_docarr_file, doc_arr)
    #
    # ifd = data.load_ifd()
    # orgn_word2wid = ifd.get_word2id()
    # assert min(wid for wid in orgn_word2wid.values()) == 0
    # new_word2wid = {word: wid + 1 for word, wid in orgn_word2wid.items()}
    # iu.dump_json(data.word2wid_file, new_word2wid)


if __name__ == '__main__':
    for d in class_list:
        print(d.name)
        Sampler(d).load()
    exit()
    mu.multi_process(transfer_files, args_list=[(c,) for c in class_list])
    # summary_datasets()
    # exit()
    # to_btm()
    # exit()
    # for _d in [Data20ng]:
    #     print('Going to process: {}, continue?'.format(_d.name))
    #     input()
    #     _d = _d()
    #     _d.filter_into_temp()
    #     _d.filter_from_temp()
    #     _d.train_word2vec(embed_dim=300)
    #     _d.train_centroids()
