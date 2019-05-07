from cqa.baselinee import J
from cqa.mee.main import Runner, reach_partition, start_pbar, get_session, Nodes


class Runner2(Runner):
    def __init__(self, args: dict):
        super(Runner2, self).__init__(args)
        self.batch_size = self.model_args[J.bs]
        self.data_multi = self.model_args[J.dmu_]

    def get_model_class(self):
        from cqa.baselinee.b1 import B1
        from cqa.baselinee.counterpart import AAAI15, AAAI17, IJCAI15
        return {v.__name__: v for v in [AAAI15, AAAI17, IJCAI15, B1]}[self.model_name]

    def build_model(self):
        len_q, len_a = self.data.d_obj.len_q, self.data.d_obj.len_a
        word_vec, user_vec = self.data.d_obj.word_vec, self.data.d_obj.user_vec
        self.model = self.get_model_class()(self.model_args)
        self.model.build(word_vec, user_vec, len_q, len_a)
        self.model.set_session(get_session(self.gpu_id, self.gpu_frac, Nodes.is_1702()))

    def iterate_data(self):
        train_num = self.data.rvs_size()[0]
        for e in range(self.epoch_num):
            train_gen = self.data.q0a0u0_q1a1u1(self.batch_size, self.data_multi)
            update_pbar, close_pbar = start_pbar(50, 'train')
            for qid, q0, a0, u0, q1, a1, u1 in train_gen:
                losses = self.model.train_step(q0, a0, u0, q1, a1, u1)
                update_pbar(qid, train_num)
                if reach_partition(qid, train_num, 4) or qid == train_num - 1:
                    self.ppp(losses)
                    if self.should_early_stop():
                        self.ppp('early stop')
                        return
            close_pbar()


if __name__ == '__main__':
    from cqa.baselinee.grid import get_args

    Runner2(get_args()).run()
