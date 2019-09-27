from clu.data.datasets import *


class Sampler:
    def __init__(self, d_cls):
        if d_cls in name2d_class:
            d_cls = name2d_class[d_cls]
        self.d_obj: Data = d_cls()
        self.word2wid: dict = dict()
        self.word2vec: dict = dict()
        self.docarr: List[Document] = list()
        self.eval_batches: List[List[Document]] = list()
        self.clu_embed_init = self.word_embed_init = None

    def load(self, embed_dim: int = 64, topic_ratio: float = 1,
             pad_len: int = 0, use_tfidf: bool = False):
        self.docarr, self.word2wid = self.d_obj.load_docarr_and_word2wid()
        if embed_dim:
            self.prepare_embedding(embed_dim, topic_ratio)
        if pad_len > 1:
            self.pad_docarr(pad_len)
        if use_tfidf:
            self.prepare_tf_tfidf()
        self.eval_batches = [pos for pos, negs in self.generate(128, 0, False)]
        # self.eval_batches = self.split_length(self.docarr, 256)

    def pad_docarr(self, seq_len):
        # inplace
        assert seq_len is not None and seq_len > 0
        for doc in self.docarr:
            n_tokens = len(doc.tokenids)
            if n_tokens == seq_len:
                doc.tokenids = doc.tokenids
            elif n_tokens < seq_len:
                doc.tokenids = doc.tokenids + [0] * (seq_len - n_tokens)
            else:
                doc.tokenids = doc.tokenids[:seq_len]

    def prepare_embedding(self, embed_dim: int, topic_ratio: float):
        self.word2vec = self.d_obj.load_word2vec(embed_dim)
        self.clu_embed_init = self.d_obj.load_clu_init(embed_dim, topic_ratio)
        self.word_embed_init = [None] * len(self.word2vec)
        for word, wid in self.word2wid.items():
            self.word_embed_init[wid - 1] = self.word2vec[word]
        for e in self.word_embed_init:
            assert e is not None
        self.word_embed_init = np.array(self.word_embed_init)

    def prepare_tf_tfidf(self):
        du.docarr_fit_tfidf(self.word2wid, self.docarr)

    def get_xy_for_stc(self, weight_a: float = 0.0001, n_pc: int = 1):
        from scipy.sparse import vstack
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.decomposition import TruncatedSVD
        # term freq
        wf = vstack([doc.tf for doc in self.docarr])
        wf = wf.sum(axis=0)
        wf = np.array(wf).reshape([1, -1]).T
        wsum = np.sum(wf) * weight_a
        pw = wsum / (wsum + wf)
        pw[0][0] = 0
        # weighted average
        padding = np.zeros([1, self.word_embed_init.shape[1]])
        weighted_embed = np.concatenate([padding, self.word_embed_init], axis=0)
        weighted_embed = pw * weighted_embed
        X = [np.mean(weighted_embed[doc.tokenids], axis=0) for doc in self.docarr]
        X = np.array(X)
        # pca
        svd = TruncatedSVD(n_components=n_pc, n_iter=10)
        svd.fit(X)
        pc = svd.components_
        # back
        X = X - X.dot(pc.T) * pc
        X = MinMaxScaler().fit_transform(X)
        print(pc.shape, X.shape)
        Y = np.array([doc.topic for doc in self.docarr])
        return X, Y

    # @staticmethod
    # def split_length(docarr: List[Document], batch_size: int) -> List[List[Document]]:
    #     docarr = sorted(docarr, key=lambda x: len(x.tokenids))
    #     batches, batch = list(), list()
    #     prev_len = len(docarr[0].tokenids)
    #     for i, doc in enumerate(docarr):
    #         doc_len = len(doc.tokenids)
    #         if doc_len != prev_len or len(batch) >= batch_size:
    #             batches.append(batch)
    #             batch = [doc]
    #         else:
    #             batch.append(doc)
    #         if i >= len(docarr) - 1:
    #             batches.append(batch)
    #             break
    #         prev_len = doc_len
    #     return batches

    def generate(self, batch_size: int, neg_batch_num: int, shuffle: bool):
        docarr = au.shuffle(self.docarr) if shuffle else self.docarr
        docarr_list = au.split_slices(docarr, batch_size)
        for arr in docarr_list:
            yield arr, None
        # batches = self.split_length(docarr, batch_size)
        # print('shuffle_generate - batch num:', len(batches))
        # for i in range(len(batches)):
        #     p_batch: List[Document] = batches[i]
        #     yield p_batch, None
        # n_idxes: List[int] = np.random.choice([j for j in i_range if j != i], neg_batch_num)
        # n_batches: List[List[Document]] = [batches[j] for j in n_idxes]
        # yield p_batch, n_batches


def summary_datasets():
    import pandas as pd
    df = pd.DataFrame(columns=['K', 'D', 'V', 'Avg-len', 'Max-len', 'Min-len'])
    for idx, cls in enumerate([DataTrec, DataGoogle, DataEvent, Data20ng, DataReuters]):
        d_obj = cls()
        print(d_obj.docarr_file)
        docarr, word2wid = d_obj.load_docarr_and_word2wid()
        len_arr = [len(d.tokenids) for d in docarr]
        K = len(set([d.topic for d in docarr]))
        D = len(docarr)
        V = len(word2wid)
        avg_len = round(float(np.mean(len_arr)), 2)
        max_len = round(float(np.max(len_arr)), 2)
        min_len = round(float(np.min(len_arr)), 2)
        df.loc[d_obj.name] = [K, D, V, avg_len, max_len, min_len]
    print(df)
    df.to_csv('summary.csv')


def transfer_files(d_class: Data):
    data = d_class()
    print(data.name)
    print(data.docarr_file)
    print(data.word2wid_file)
    print('train word2vec & centroids')
    print()
    for dim in [16, 32]:
        for ratio in [1]:
            print(data.name, dim, ratio)
            data.train_word2vec(dim)
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


def main():
    s = Sampler(DataTrec)
    s.load(64, 1, 0, True)
    s.get_xy_for_stc()
    return

    from utils import au
    import pandas as pd
    df = pd.DataFrame()
    i = 0
    for dc in class_list:
        for dim in dc().support_dims:
            smp = Sampler(dc)
            smp.load(dim, 1, 0, False)
            w2v = smp.word2vec
            pc_true = [doc.topic for doc in smp.docarr]
            d_mean = [np.mean([w2v[w] for w in doc.tokens], axis=0) for doc in smp.docarr]
            d_mean = np.array(d_mean)
            c_emb = smp.clu_embed_init
            pc_score = np.matmul(d_mean, c_emb.T)
            pc_pred = np.argmax(pc_score, axis=1).reshape(-1)
            kv = {'data': dc.name, 'embed_dim': dim}
            scores = au.scores(pc_true, pc_pred)
            kv.update(scores)
            for k, v in kv.items():
                df.loc[i, k] = v
            i += 1
    df.to_csv('no_train.csv')


if __name__ == '__main__':
    main()
