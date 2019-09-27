import numpy as np
from utils import iu, au
from typing import Dict, List
import uclu.data.document as udoc
from scipy.sparse import vstack


class DataSo:
    base = '/home/cdong/works/uclu/data'
    name = 'so'
    topic_num = 111

    def __init__(self):
        self.docarr_file = self.fill('docarr.json')
        self.word2wint_file = self.fill('word2wint.json')
        self.user2uint_file = self.fill('user2uint.json')
        self.tf_file = self.fill('tf_matrix.mtx')
        self.tfidf_file = self.fill('tfidf_matrix.mtx')
        self.support_dims = [2 ** i for i in range(4, 9)] + [300]

    def fill(self, files):
        if isinstance(files, list):
            return [iu.join(self.base, self.name, f) for f in files]
        elif isinstance(files, str):
            return iu.join(self.base, self.name, files)

    def load_docarr(self) -> List[udoc.Document]:
        return udoc.load_docarr(self.docarr_file)

    def load_word2wint(self) -> Dict:
        return iu.load_json(self.word2wint_file)

    def load_user2uint(self) -> Dict:
        return iu.load_json(self.user2uint_file)

    @staticmethod
    def check_path(file: str, required: bool):
        if required and not iu.exists(file):
            raise ValueError('%s is required but does not exist' % file)

    def get_word2vec_file(self, dim: int, required=True) -> str:
        file = iu.join(self.base, self.name, 'embeddings', f'{self.name}_w2v_{dim}.pkl')
        self.check_path(file, required)
        return file

    def get_user2vec_file(self, dim: int, required=True) -> str:
        file = iu.join(self.base, self.name, 'embeddings', f'{self.name}_u2v_{dim}.pkl')
        self.check_path(file, required)
        return file

    def get_clu_init_file(self, dim: int, topic_ratio: float, required=True) -> str:
        fname = '{}_clu_{}_{:.2f}.pkl'.format(self.name, dim, topic_ratio)
        file = iu.join(self.base, self.name, 'embeddings', fname)
        self.check_path(file, required)
        return file

    def load_word2vec(self, dim: int) -> Dict[str, np.ndarray]:
        return iu.load_pickle(self.get_word2vec_file(dim))

    def load_user2vec(self, dim: int) -> Dict[str, np.ndarray]:
        return iu.load_pickle(self.get_user2vec_file(dim))

    def load_clu_init(self, dim: int, topic_ratio: float = 1) -> np.ndarray:
        return iu.load_pickle(self.get_clu_init_file(dim, topic_ratio))


class_list = [DataSo]
name_list = [c.name for c in class_list]
name2d_class = dict(zip(name_list, class_list))


class Sampler:
    def __init__(self, d_class):
        if d_class in name2d_class:
            d_class = name2d_class[d_class]
        self.d_obj = d_class()
        self.vocab_min: int = None
        self.vocab_size: int = None
        self.user_min: int = None
        self.user_size: int = None

        self.word2wint: dict = dict()
        self.user2uint: dict = dict()
        self.word2vec: dict = dict()
        self.user2vec: dict = dict()
        self.docarr: List[udoc.Document] = list()
        self.eval_batches: List[List[udoc.Document]] = list()
        self.w_embed = self.u_embed = self.c_embed = None

    def load(self, dim: int, tpad: int, bpad: int, topic_ratio: float = 1):
        self.load_basic()
        if tpad or bpad:
            self.pad_docarr(tpad, bpad)
        if dim and topic_ratio:
            self.prepare_embedding(dim, topic_ratio)
        self.eval_batches = list(self.generate(256, False))

    def generate(self, batch_size: int, shuffle: bool):
        docarr = au.shuffle(self.docarr) if shuffle else self.docarr
        docarr_list = au.split_slices(docarr, batch_size)
        for arr in docarr_list:
            yield arr

    def load_basic(self):
        self.docarr = self.d_obj.load_docarr()
        self.word2wint = self.d_obj.load_word2wint()
        self.user2uint = self.d_obj.load_user2uint()
        self.vocab_min = min(self.word2wint.values())
        self.vocab_size = len(self.word2wint)
        self.user_min = min(self.user2uint.values())
        self.user_size = len(self.user2uint)

    def pad_docarr(self, tpad, bpad):
        for doc in self.docarr:
            title_gen = iter(doc.title)
            body_gen = doc.flat_body()
            doc.tseq = [next(title_gen, 0) for _ in range(tpad)] if tpad else doc.title
            doc.bseq = [next(body_gen, 0) for _ in range(bpad)] if bpad else list(body_gen)

    def fit_sparse(self, add_body: bool, tfidf: bool, p_num: int = 12):
        self.fit_tf(add_body, p_num)
        if tfidf:
            self.fit_tf_idf()

    def fit_tf(self, add_body: bool, p_num: int):
        udoc.docarr_fit_tf_multi(self.docarr, self.word2wint, add_body, p_num)

    def fit_tf_idf(self):
        udoc.docarr_fit_tfidf(self.docarr)

    # def fit_save_sparse(self):
    #     self.load_basic()
    #     self.fit_sparse(p_num=20)
    #     tf = self.get_feature('tf')
    #     tfidf = self.get_feature('tfidf')
    #     iu.dump_pickle(self.d_obj.tf_file, tf)
    #     iu.dump_pickle(self.d_obj.tfidf_file, tfidf)
    #     # io.mmwrite(self.d_obj.tf_file, tf)
    #     # io.mmwrite(self.d_obj.tfidf_file, tfidf)

    # def load_sparse(self, to_dense: bool = False):
    #     tf_matrix = iu.load_pickle(self.d_obj.tf_file)
    #     tfidf_matrix = iu.load_pickle(self.d_obj.tfidf_file)
    #     # tf_matrix = io.mmread(self.d_obj.tf_file)
    #     # tfidf_matrix = io.mmread(self.d_obj.tfidf_file)
    #     if to_dense:
    #         tf_matrix = tf_matrix.todense()
    #         tfidf_matrix = tfidf_matrix.todense()
    #     else:
    #         tf_matrix = tf_matrix.tocsr()
    #         tfidf_matrix = tfidf_matrix.tocsr()
    #     for tf, tfidf, doc in zip(tf_matrix, tfidf_matrix, self.docarr):
    #         doc.tf = tf
    #         doc.tfidf = tfidf
    #     print('load sparse over,', tf_matrix.shape, tfidf_matrix.shape, len(self.docarr))

    def get_feature(self, using):
        vs = np.vstack if isinstance(self.docarr[0].tf, np.ndarray) else vstack
        if using == 'tf':
            stacked = vs([doc.tf for doc in self.docarr])
        elif using == 'tfidf':
            stacked = vs([doc.tfidf for doc in self.docarr])
        else:
            raise ValueError('not supported feature:', using)
        return stacked

    def prepare_embedding(self, dim: int, topic_ratio: float):
        self.word2vec = self.d_obj.load_word2vec(dim)
        self.user2vec = self.d_obj.load_user2vec(dim)
        self.w_embed = self.get_embed(self.word2wint, self.word2vec)
        self.u_embed = self.get_embed(self.user2uint, self.user2vec)
        self.c_embed = self.d_obj.load_clu_init(dim, topic_ratio)

    @staticmethod
    def get_embed(w2i: dict, w2v: dict):
        print('get_embed,  word num:{}, min index:{}'.format(len(w2i), min(w2i.values())))
        reverse: int = min(w2i.values())
        params: List = [None] * len(w2i)
        for k, i in w2i.items():
            params[i - reverse] = w2v[k]
        padding = np.zeros((reverse, len(params[0])))
        embed = np.concatenate([padding, params], axis=0)
        return embed

    def get_xy_for_stc(self, add_body: bool, weight_a: float = 0.001, n_pc: int = 1):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.decomposition import TruncatedSVD
        # term freq
        wf = self.get_feature('tf').sum(axis=0).reshape([-1, 1])
        pw = weight_a / (weight_a + np.array(wf) / np.sum(wf))
        pw[:, :min(self.word2wint.values())] = 0
        # weighted average
        weighted_embed = pw * self.w_embed
        X = []
        for doc in self.docarr:
            wints = list(doc.title)
            if add_body:
                wints += list(doc.flat_body())
            X.append(np.mean(weighted_embed[wints], axis=0))
        X = np.array(X)
        # pca
        svd = TruncatedSVD(n_components=n_pc, n_iter=10)
        svd.fit(X)
        pc = svd.components_
        # back
        X = X - X.dot(pc.T) * pc
        X = MinMaxScaler().fit_transform(X)
        Y = np.array(au.reindex([doc.tag for doc in self.docarr]))
        return X, Y


if __name__ == '__main__':
    s = Sampler(DataSo)
    s.load_basic()
    cnt = 0
    for doc in s.docarr:
        doc.split_texts()
        cnt += len(doc.all_texts)
    print(cnt, len(s.docarr))

    # s.prepare_embedding(16, 1)
    # s.fit_tf(add_body=False, p_num=10)
    # s.get_xy_for_stc(add_body=True)

    # s.fit_save_sparse()
    # s.load_basic()
    # s.load_sparse()
    # s.load(0, 0, 0, 0)
    # print(len(s.docarr), len(s.word2vec), len(s.word2wint))
