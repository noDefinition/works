from cqa.mee.main import Runner, reach_partition, start_pbar
from cqa.baselinee import *


class Runner2(Runner):
    def __init__(self, args: dict):
        super(Runner2, self).__init__(args)
        self.batch_size = self.args[bs_]
        self.data_multi = self.args[dmu_]

    def get_model_class(self):
        from cqa.baselinee.b1 import B1
        return B1

    def iterate(self):
        train_num = len(self.train_data)
        train_gen = self.sampler.a0u0a1u1(self.batch_size, self.data_multi)
        for e in range(self.epoch_num):
            update_pbar, close_pbar = start_pbar(50, 'train')
            for bid, (a0, u0, a1, u1) in enumerate(train_gen):
                self.model.train_step(self.sess, a0, u0, a1, u1)
                update_pbar(bid, train_num)
                if reach_partition(bid, train_num, 4) or bid == train_num - 1:
                    if self.should_early_stop():
                        return
            close_pbar()


# def main():
#     args = get_args()
#     logger_file, writer_file = get_logger_writer_file(args, {'lg'})
#     # entries = [(k, v) for k, v in args.__dict__.items() if v is not None]
#     # name = au.entries2name(entries, exclude={'lg'}, postfix='.txt')
#     # writer_file = iu.join(args.lg, 'gid={}'.format(args.gid))
#     # logger_file = iu.join(args.lg, name)
#
#     adw = AdhocWriter2()
#     adw.set_early_stop(args.es)
#     adw.set_logger(logger_file)
#
#     sampler = Sampler(args.dn)
#     sampler.load_parts(args.fda)
#     valid_data = sampler.get_valid()
#     test_data = sampler.get_test()
#     r, _, _ = sampler.rvs_size()
#     train_num = r
#
#     model = Pair(args)
#     model.build(sampler.d_obj.word_vec, sampler.d_obj.user_vec)
#     sess = get_session(args.gi, args.gp, run_init=True)
#
#     step = 100
#     for e in range(args.ep):
#         adw.ppp('\nepoch:{}'.format(e))
#         with tqdm(total=step, ncols=80, leave=True, desc='train') as pbar:
#             for bid, (a0, u0, a1, u1) in enumerate(sampler.a0u0a1u1(args.bs)):
#                 if bid > 0 and bid % (train_num // step) == 0:
#                     pbar.update(1)
#                 loss = model.train_step(sess, a0, u0, a1, u1)
#                 if bid > 0 and bid % (train_num // 5) == 0:
#                     print('\nbid:{} loss:{:.5f}'.format(bid, loss))
#                     valid_mrs = evaluate(sess, model, valid_data, 'valid')
#                     adw.ppp(iu.dumps({'vali': valid_mrs}))
#                     if adw.compare_and_judge(valid_mrs):
#                         test_mrs = evaluate(sess, model, test_data, 'test')
#                         adw.ppp(iu.dumps({'test': test_mrs}))
#                         return


if __name__ == '__main__':
    from cqa.baselinee.grid import get_args

    Runner2(get_args()).run()
