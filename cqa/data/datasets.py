import numpy as np
from utils import au, iu, tmu

RVS = R, V, S = 'train', 'valid', 'test'


def partition_qids(aid2qu_file, seed, pidx):
    class User:
        lookup = dict()

        def __init__(self, uid):
            self.uid = uid
            self.qs = list()
            self.rvs2qs = None

        def add_q(self, q):
            self.qs.append(q)

        def split_rvs_qs(self):
            self.rvs2qs = dict((x, list()) for x in RVS)
            for q in self.qs:
                self.rvs2qs[q.rvs].append(q)
            return self.rvs2qs

        @staticmethod
        def find(uid):
            if uid not in User.lookup:
                User.lookup[uid] = User(uid)
            return User.lookup[uid]

    class Ques:
        lookup = dict()

        def __init__(self, qid):
            self.qid = qid
            self.rvs = np.random.choice(RVS, p=[0.78, 0.11, 0.11])

        def set_rvs(self, x):
            # assert x in RVS
            self.rvs = x

        @staticmethod
        def find(qid):
            if qid not in Ques.lookup:
                Ques.lookup[qid] = Ques(qid)
            return Ques.lookup[qid]

    np.random.seed(seed)
    for aid, (qid, uid) in iu.load_pickle(aid2qu_file).items():
        ques = Ques.find(qid)
        user = User.find(uid)
        ques.add_user(user)
        user.add_q(ques)
    for i in range(10000):
        print(i, 'th trial')
        for uid, user in User.lookup.items():
            rvs_quess = user.split_rvs_qs()
            rc, vc, sc = [len(rvs_quess[x]) for x in RVS]
            if rc < 3:
                print('recheck on ', uid)
                if vc > 0:
                    user.rvs2quess[V][0].set_rvs(R)
                elif sc > 0:
                    user.rvs2quess[S][0].set_rvs(R)
        can = True
        for uid, user in User.lookup.items():
            rvs_quess = user.split_rvs_qs()
            counts = rc, vc, sc = [len(rvs_quess[x]) for x in RVS]
            if rc < 3 or vc < 1 or sc < 1:
                print(pidx, counts)
                can = False
                break
        if can:
            # print(pidx, 'done')
            rvs2qid = dict((x, list()) for x in RVS)
            for qid, ques in Ques.lookup.items():
                rvs2qid[ques.rvs].append(qid)
            lenarr = [len(qids) for x, qids in rvs2qid.items()]
            print(lenarr, [s / sum(lenarr) for s in lenarr])

            all_qids = list(Ques.lookup.keys())
            rvs_qids = au.merge(rvs2qid.values())
            print(len(all_qids), len(set(all_qids)), '; ', len(rvs_qids), len(set(rvs_qids)))
            assert len(all_qids) == len(rvs_qids) and set(all_qids) == set(rvs_qids)
            print('valid partition found')
            return rvs2qid


class _Data:
    len_q = len_a = name = None

    def __init__(self):
        f = self.fill
        self.user_vec_file = f('user_vec_64d.pkl')
        self.word_vec_file = f('word_vec_64d.pkl')
        self.qid2qauv_file = f('cdong', 'qid2qauv_dict.pkl')
        self.rvs2qids_file = f('cdong', 'rvs2qids_dict.pkl')
        # self.simple_qid2qauv_file = f('simple', 'qid2qauv_dict.pkl')
        # self.simple_rvs2qids_file = f('simple', 'rvs2qids_dict.pkl')
        self.need_transfer = [
            self.user_vec_file, self.word_vec_file, self.qid2qauv_file, self.rvs2qids_file,
            # self.simple_qid2qauv_file, self.simple_rvs2qids_file
        ]
        self.qid2qauv = self.rvs2qids = self.user_vec = self.word_vec = None
        self.aid2votes = self.aid2pf16 = self.user_vec_bert = self.qid2auv_bert = None

    def fill(self, *names):
        cdong_home = '/home/cdong/works/cqa/data/{}'.format(self.name)
        return iu.join(cdong_home, *names)

    # @tmu.stat_time_elapse
    # def init_save_rvs2qids_qid2qauv(self):
    #     rvs2qids = partition_qids(self.aid2qu_file, 42987, 0)
    #     self.load_wwang_parts()
    #
    #     uid_sort = sorted(set(uid for aid, (qid, uid) in self.aid2qu.items()))
    #     uid2uint = dict((uid, uint) for uint, uid in enumerate(uid_sort))
    #
    #     # user vec init by averaging
    #     _, w_dim = self.word_vec.shape
    #     uint2v_sum = np.zeros(shape=(len(uid2uint), w_dim), dtype=np.float32)
    #     uint2count = np.zeros(shape=(len(uid2uint), 1), dtype=np.float32)
    #     for aid, (qid, uid) in self.aid2qu.items():
    #         uint = uid2uint[uid]
    #         w_vecs = [self.word_vec[wid - 1] for wid in self.aid2wids[aid] if wid > 0]
    #         uint2v_sum[uint] += np.mean(w_vecs, axis=0)
    #         uint2count[uint][0] += 1
    #     uint2vec = uint2v_sum / uint2count
    #
    #     qid2auvs = {}
    #     for aid, (qid, uid) in self.aid2qu.items():
    #         qid2auvs.setdefault(qid, list()).append((aid, uid, self.aid2vote[aid]))
    #     qid2qauv = {}
    #     for idx, (qid, auvs) in enumerate(qid2auvs.items()):
    #         auvs = sorted(auvs, key=lambda x: x[-1], reverse=True)
    #         aids, uids, votes = au.transpose(auvs)
    #         qwid = self.qid2wids[qid]
    #         awids = [self.aid2wids[aid] for aid in aids]
    #         uints = [uid2uint[uid] for uid in uids]
    #         qid2qauv[qid] = (qwid, awids, uints, votes)
    #         if idx < 10:
    #             print(auvs)
    #             print()
    #             print(aids)
    #             print(qwid)
    #             print(qid2qauv[qid][1])
    #             print(qid2qauv[qid][2])
    #             print('-------------------------------')
    #
    #     iu.dump_pickle(self.user_vec_file, uint2vec)
    #     iu.dump_pickle(self.uid2uint_file, uid2uint)
    #     iu.dump_pickle(self.rvs2qids_file, rvs2qids)
    #     iu.dump_pickle(self.qid2qauv_file, qid2qauv)

    # @tmu.stat_time_elapse
    # def init_save_rvs2qids_qid2qauv_simple(self):
    #     tmu.check_time('load')
    #     print(self.qid2qauv_file)
    #     self.rvs2qids = iu.load_pickle(self.rvs2qids_file)
    #     self.qid2qauv = iu.load_pickle(self.qid2qauv_file)
    #     tmu.check_time('load')
    #     rvs2lim = {R: 1000, V: 100, S: 100}
    #     rvs2qids_simple = {x: np.random.choice(qids, rvs2lim[x], replace=False)
    #                        for x, qids in self.rvs2qids.items()}
    #     chosen_qids = set(au.merge(rvs2qids_simple.values()))
    #     print('chosen qid num', len(chosen_qids), len(au.merge(rvs2qids_simple.values())))
    #     qid2qauv_simple = {qid: self.qid2qauv[qid] for qid in chosen_qids}
    #     iu.dump_pickle(self.simple_rvs2qids_file, rvs2qids_simple)
    #     iu.dump_pickle(self.simple_qid2qauv_file, qid2qauv_simple)

    def load_word_vec(self):
        self.word_vec = iu.load_pickle(self.word_vec_file)

    def load_user_vec(self):
        self.user_vec = iu.load_pickle(self.user_vec_file)

    """ bert """

    def load_user_vec_bert(self):
        file = self.fill('bert', 'user_vec_bert.pkl')
        self.user_vec_bert = iu.load_pickle(file)

    @tmu.stat_time_elapse
    def get_qid2auv_bert(self):
        qau_list = iu.load_pickle(self.fill('_mid', 'qau_list.pkl'))
        uid2uint = iu.load_pickle(self.fill('_mid', 'uid2uint_dict.pkl'))
        aid2pf16 = iu.load_pickle(self.fill('bert', 'aid2pf16.pkl'))
        aid2vote = iu.load_pickle(self.fill('aid2vote_dict.pkl'))
        print(len(qau_list), len(uid2uint), len(aid2pf16), len(aid2vote))
        self.qid2auv_bert = dict()
        for qid, aid, uid in qau_list:
            if qid not in self.qid2auv_bert:
                self.qid2auv_bert[qid] = [], [], []
            al, ul, vl = self.qid2auv_bert[qid]
            al.append(aid2pf16[aid])
            ul.append(uid2uint[uid])
            vl.append(aid2vote[aid])

    def load_bert_full(self):
        self.load_user_vec_bert()
        self.get_qid2auv_bert()
        self.rvs2qids = iu.load_pickle(self.rvs2qids_file)

    """ normal """

    @tmu.stat_time_elapse
    def load_cdong_full(self):
        self.load_word_vec()
        self.load_user_vec()
        self.rvs2qids = iu.load_pickle(self.rvs2qids_file)
        self.qid2qauv = iu.load_pickle(self.qid2qauv_file)

    # @tmu.stat_time_elapse
    # def load_cdong_simple(self):
    #     self.load_word_vec()
    #     self.load_user_vec()
    #     self.rvs2qids = iu.load_pickle(self.simple_rvs2qids_file)
    #     self.qid2qauv = iu.load_pickle(self.simple_qid2qauv_file)

    def rvs_size(self):
        return [len(self.rvs2qids[x]) for x in RVS]

    def load(self, is_full_data):
        self.load_cdong_full() if bool(is_full_data) else self.load_cdong_simple()
        print('word_vec.shape:{},  user_vec.shape:{}'.format(
            self.word_vec.shape, self.user_vec.shape),
            '| r:{}, v:{}, s:{}'.format(*self.rvs_size()))
        for qid, (ql, al, ul, vl) in self.qid2qauv.items():
            v = ql[:self.len_q]
            if len(v) < self.len_q:
                ql = v + [0] * (self.len_q - len(v))
            self.qid2qauv[qid] = (ql, al, ul, vl)

    def get_train_qids(self):
        return self.rvs2qids[R]

    def get_valid_qids(self):
        return self.rvs2qids[V]

    def get_test_qids(self):
        return self.rvs2qids[S]

    def get_auv_bert(self, qid):
        return self.qid2auv_bert[qid]

    def gen(self, rvs, source: dict, shuffle: bool):
        qids = self.rvs2qids[rvs]
        if shuffle:
            qids = au.shuffle(qids)
        for qid in qids:
            yield source[qid]

    def gen_train(self, shuffle=True):
        return self.gen(R, self.qid2qauv, shuffle=shuffle)

    def gen_valid(self):
        return self.gen(V, self.qid2qauv, shuffle=False)

    def gen_test(self):
        return self.gen(S, self.qid2qauv, shuffle=False)


class DataSo(_Data):
    # ques: 139128, answ: 884261, user: 40213
    # word: 52457(w/o padding)
    name = 'so'
    len_q = 20
    len_a = 200

    def __init__(self):
        super(DataSo, self).__init__()
        f = self.fill
        self.aid2wids_file = f('aid2wids_dict.pkl')
        self.aid2qu_file = f('aid2qu_dict.pkl')
        self.qid2wids_file = f('qid2wids_dict.pkl')
        self.aid2vote_file = f('aid2vote_dict.pkl')
        self.uid2uint_file = f('uid2uint_dict.pkl')
        self.qid2wids = self.aid2wids = self.aid2vote = self.aid2qu = None

    @tmu.stat_time_elapse
    def load_wwang_parts(self):
        self.load_word_vec()
        self.qid2wids = iu.load_pickle(self.qid2wids_file)
        self.aid2wids = iu.load_pickle(self.aid2wids_file)
        print('load awids/qwids over')
        self.aid2vote = iu.load_pickle(self.aid2vote_file)
        self.aid2qu = iu.load_pickle(self.aid2qu_file)
        assert len(self.aid2vote) == len(self.aid2wids) == len(self.aid2qu)
        print('load qid/aid/uid over'.format(self.name))

    # def transfer_in(self):
    #     h = '/home/wwang/Projects/QAR_data/{}_data'.format(self.name)
    #     iu.copy(h + '/features/wordVec_64d.pkl', self.word_vec_file)
    #     iu.copy(h + '/features/qid2wids_dict.pkl', self.qid2wids_file)
    #     iu.copy(h + '/features/aid2wids_dict.pkl', self.aid2wids_file)
    #     iu.copy(h + '/raw/aid2vote_dict.pkl', self.aid2vote_file)
    #     iu.copy(h + '/data/aid2qu_dict.pkl', self.aid2qu_file)


class DataZh(_Data):
    # ques: 52555, answ: 2671359, user: 168088
    # word: 65895
    name = 'zh'
    len_q = 40
    len_a = 200


class_list = [DataSo, DataZh]
name_list = [_c.name for _c in class_list]
name2d_class = dict(zip(name_list, class_list))

if __name__ == '__main__':
    from utils.node_utils import Nodes
    from clu.data.remote_transfer import transfer_files, OUT

    dict().copy()
    files = DataSo().need_transfer + DataZh().need_transfer
    transfer_files(files, files, OUT, Nodes.alias_gpu)
