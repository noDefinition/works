from typing import Dict, List

import numpy as np
from scipy.sparse import vstack

import uclu.data.document as udoc
from utils import au, iu


class Data:
    base = '/home/cdong/works/uclu/data'
    support_dims = [2 ** i for i in range(4, 9)] + [300]
    name = topic_num = None

    def __init__(self):
        self.docarr_file = self.fill('docarr.json')
        self.word2wint_file = self.fill('word2wint.json')
        self.user2uint_file = self.fill('user2uint.json')

    def __call__(self, *args, **kwargs):
        return self

    def fill(self, *files):
        return iu.join(self.base, self.name, *files)

    def path_lv(self, lv: int):
        return self.fill('questions_lv{}'.format(lv))

    def list_lv(self, lv: int):
        path = self.path_lv(lv)
        if not iu.exists(path):
            iu.mkdir(path)
        return iu.list_children(path, ctype=iu.FILE, pattern='^que', full_path=True)

    def load_docarr(self) -> List[udoc.Document]:
        return udoc.load_docarr(self.docarr_file)

    def load_word2wint(self) -> Dict[str, int]:
        return iu.load_json(self.word2wint_file)

    def load_user2uint(self) -> Dict[str, int]:
        return iu.load_json(self.user2uint_file)

    def get_word2vec_file(self, dim: int) -> str:
        file = iu.join(self.base, self.name, 'embeddings', f'{self.name}_w2v_{dim}.pkl')
        return file

    def get_user2vec_file(self, dim: int) -> str:
        file = iu.join(self.base, self.name, 'embeddings', f'{self.name}_u2v_{dim}.pkl')
        return file

    def get_clu_init_file(self, dim: int, topic_ratio: float) -> str:
        fname = '{}_clu_{}_{:.2f}.pkl'.format(self.name, dim, topic_ratio)
        file = iu.join(self.base, self.name, 'embeddings', fname)
        return file

    def load_word2vec(self, dim: int) -> Dict[str, np.ndarray]:
        return iu.load_pickle(self.get_word2vec_file(dim))

    def load_user2vec(self, dim: int) -> Dict[str, np.ndarray]:
        return iu.load_pickle(self.get_user2vec_file(dim))

    def load_clu_init(self, dim: int, topic_ratio: float = 1) -> np.ndarray:
        return iu.load_pickle(self.get_clu_init_file(dim, topic_ratio))

    def stats(self):
        docarr = self.load_docarr()
        tagset, uintset, wordset = set(), set(), set()
        title_sum = body_sum = 0
        title_max = body_max = 0
        title_min = body_min = 1 << 31
        for doc in docarr:
            tagset.add(doc.tag)
            uintset.add(doc.uint)
            body = list(doc.flatten_body())
            wordset.update(set(doc.title))
            wordset.update(set(body))

            lt = len(doc.title)
            title_sum += lt
            title_max = max(title_max, lt)
            title_min = min(title_min, lt)

            lb = len(body)
            body_sum += lb
            body_max = max(body_max, lb)
            body_min = min(body_min, lb)

        print(self.name)
        print('标题')
        print('    最小长度:', title_min)
        print('    最大长度:', title_max)
        print('    平均长度:', round(title_sum / len(docarr), 2))
        print('正文')
        print('    最小长度:', body_min)
        print('    最大长度:', body_max)
        print('    平均长度:', round(body_sum / len(docarr), 2))
        print('词汇数:', len(wordset))
        print('用户数:', len(uintset))
        print('文档数:', len(docarr))
        print('标签数:', len(tagset))
        print()


class DataSo(Data):
    name = 'so'
    topic_num = 111


class DataSf(Data):
    name = 'sf'
    topic_num = 61


class DataAu(Data):
    name = 'au'
    topic_num = 73


class DataSoB(Data):
    name = 'sob'
    topic_num = 111


class DataZh(Data):
    name = 'zh'
    topic_num = 181


data_so = DataSo()
data_sf = DataSf()
data_au = DataAu()
data_sob = DataSoB()

_class_list = [DataSo, DataSf, DataAu, DataZh]
_name_list = [c.name for c in _class_list]
name2d_class = dict(zip(_name_list, _class_list))


class Sampler:
    def __init__(self, d_class):
        self.d_obj = name2d_class.get(d_class, d_class)()
        self.vocab_min: int
        self.vocab_size: int
        self.user_min: int
        self.user_size: int
        self.word2wint: dict = dict()
        self.user2uint: dict = dict()
        self.word2vec: dict = dict()
        self.user2vec: dict = dict()
        self.w_embed: np.ndarray
        self.u_embed: np.ndarray
        self.c_embed: np.ndarray
        self.docarr: List[udoc.Document] = list()
        self.eval_batches: List[List[udoc.Document]] = list()

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
        self.vocab_size = max(self.word2wint.values()) + 1
        self.user_min = min(self.user2uint.values())
        self.user_size = max(self.user2uint.values()) + 1

    def pad_docarr(self, tpad, bpad):
        for doc in self.docarr:
            title_gen = iter(doc.title)
            body_gen = doc.flatten_body()
            # doc.tseq = doc.title[:tpad]
            # doc.bseq = list(body_gen)[:bpad]
            doc.tseq = [next(title_gen, 0) for _ in range(tpad)] if tpad else list(doc.title)
            doc.bseq = [next(body_gen, 0) for _ in range(bpad)] if bpad else list(body_gen)

    def fit_sparse(self, add_body: bool, tfidf: bool, p_num: int = 12):
        self.fit_tf(add_body, p_num)
        if tfidf:
            self.fit_tf_idf()

    def fit_tf(self, add_body: bool, p_num: int):  # inplace
        udoc.docarr_fit_tf_multi(self.docarr, self.word2wint, add_body, p_num)

    def fit_tf_idf(self):
        udoc.docarr_fit_tfidf(self.docarr)

    def get_feature(self, using):
        """ assert tf & tfidf already fit """
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
        self.c_embed = self.d_obj.load_clu_init(dim, topic_ratio)
        self.w_embed = self.get_embed(self.word2wint, self.word2vec)
        self.u_embed = self.get_embed(self.user2uint, self.user2vec)

    @staticmethod
    def get_embed(w2i: dict, w2v: dict) -> np.ndarray:
        print('get_embed,  vector num:{}, min index:{}'.format(len(w2i), min(w2i.values())))
        reserve: int = min(w2i.values())
        params: List = [None] * len(w2i)
        for k, i in w2i.items():
            params[i - reserve] = w2v[k]
        padding = np.zeros((reserve, len(params[0])))
        embed = np.concatenate([padding, params], axis=0)
        return embed

    def get_xy_for_stc(self, add_body: bool, weight_a: float = 0.001, n_pc: int = 1):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.decomposition import TruncatedSVD, PCA
        # term freq
        self.load_basic()
        tf, _, Y = iu.load_pickle(self.get_tf_idf_file(add_body))
        # Y = np.array(au.reindex([doc.tag for doc in self.docarr]))
        # wf = self.get_feature('tf').sum(axis=0).reshape([-1, 1])
        wf = tf.sum(axis=0).reshape([-1, 1])
        pw = weight_a / (weight_a + np.array(wf) / np.sum(wf))
        pw[:, :min(self.word2wint.values())] = 0
        # weighted average
        weighted_embed = pw * self.w_embed
        X = []
        for doc in self.docarr:
            wints = list(doc.title)
            if add_body:
                wints += list(doc.flatten_body())
            X.append(np.mean(weighted_embed[wints], axis=0))
        X = np.array(X)
        # decomposition
        svd = TruncatedSVD(n_components=n_pc, n_iter=10)
        # svd = PCA(n_components=n_pc)
        svd.fit(X)
        pc = svd.components_
        # back
        X = X - X.dot(pc.T) * pc
        X = MinMaxScaler().fit_transform(X)
        return X, Y

    def get_tf_idf_file(self, add_body: bool):
        add_body_str = ['nob', 'addb'][int(add_body)]
        return self.d_obj.fill('tf_tfidf_tags_{}.pkl'.format(add_body_str))

    def dump_tf_tfidf_tags(self, add_body: bool):
        self.load_basic()
        self.fit_sparse(add_body, tfidf=True, p_num=10)
        tf = vstack([doc.tf for doc in self.docarr])
        tfidf = vstack([doc.tfidf for doc in self.docarr])
        tags = np.array(au.reindex([doc.tag for doc in self.docarr]))
        iu.dump_pickle(self.get_tf_idf_file(add_body), [tf, tfidf, tags], parents=True)

    def load_tf_tfidf_tags(self, add_body):
        tf, tfidf, tags = iu.load_pickle(self.get_tf_idf_file(add_body))
        return tf, tfidf, tags


if __name__ == '__main__':
    # from utils import mu
    # def stat_data(_d_class):
    #     _d_class().stats()
    # mu.multi_process(stat_data, [(_c,) for _c in _class_list])

    # for _d_class in [DataZh]:
    #     Sampler(_d_class).dump_tf_tfidf_tags(add_body=True)
    # exit()

    for _d_class in [DataSo, DataZh]:
        _d_class().stats()

    # s = Sampler(DataSo)
    # s.load(16, 0, 0, 1)
    # for doc in s.docarr:
    #     if not doc.uint < s.user_size:
    #         print(doc.uint)
