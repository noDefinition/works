import argparse
import os

# import solver
# import utils
from prev.utils import log_dir, logger

log = logger.log
from sklearn.model_selection import ParameterGrid
from pprint import pformat
from prev import models, dataset
import atexit


class Run:
    def __init__(self):
        parser = argparse.ArgumentParser(description='models_prev')
        _ = parser.add_argument
        _('-model', type=str, default='BasicModel')
        _('-ds', '--dataset', type=str, default='test')
        # self.add_arg('-rt', '--run_times', type = int, default = 1)
        _('-am', '--arg_mode', type=str, default='fix')
        _('-msg', '--massage', type=str, default='')

        _('-summ', '--run_summary', type=int, default=1)
        _('-test', '--run_test', type=int, default=0)
        # self.add_arg('-tb', '--run_tensorboard', type = int, default = 1)

        _('-es', '--early_stop', type=int, default=5)
        _('-mee', '--max_epochs', type=int, default=100)
        _('-gpu', type=str, default='3')

        self.args = parser.parse_args()
        self.run()

    def set_logfn(self):
        if self.args.massage:
            logfn = '{} {}'.format(logfn, self.args.massage)
        else:
            logfn = '{} {}'.format(self.args.model, self.args.dataset)
        logger.set_fn(home=log_dir, fn=logfn)

    def log_msg(self):
        log('solve model: %s' % self.args.model, red=True)
        log('dataset: {}'.format(self.args.dataset), red=True)
        log('gpu: {}'.format(self.args.gpu), red=True)
        log(pformat(self.model_args_grid))

    # def summary(self):
    #     if len(self.history) > 1:
    #         log('\n\n{} summary begin!'.format(self.args.model), red=True)
    #         for h in self.history:
    #             log(h)
    #         log('best performance:', red=True)
    #         log(self.best_msg)
    #         log('summary end!\n\n', red=True)
    #     log('log file: {}\nnow: {}'.format(logger.fn, logger.date()))
    #     print('good log:', self.model.good_log)
    #     if self.model.good_log == False:
    #         os.system('rm "{}"'.format(logger.fn))
    #     print('\n\n')

    def solve(self):
        best_result = None
        self.history = []
        atexit.register(self.summary)
        self.best_msg = 'None'
        search_args = list(self.search_args())
        n = len(search_args)
        for i, model_args in enumerate(search_args):
            i += 1
            prt_args = self.prt_args(model_args)
            log('\n#args: {}/{} {}'.format(i, n, prt_args), red=True)
            result = self.model.fit(model_args, i, n)
            msg = 'result: {}, args: {}'.format(result, prt_args)
            log(msg, red=True)
            self.history.append(msg)
            if result.is_better_than(best_result):
                best_result = result
                self.best_msg = msg
        log(best_result.prt())
        atexit.unregister(self.summary)
        self.summary()

    def prt_args(self, model_args):
        ret = {}
        for k in model_args:
            if len(self.model_args_grid[k]) > 1:
                ret[k] = model_args[k]
        return ret

    def run(self, l=True):
        if l:
            dss = self.args.dataset.split(',')
            ms = self.args.model.split(',')
            for ds in dss:
                for m in ms:
                    self.args.dataset = ds
                    self.args.model = m
                    self.run(False)
            return
        Model = vars(models)[self.args.model]
        self.model_args_grid = self.get_args(Model)
        self.set_logfn()
        self.log_msg()
        self.model = Model(**self.args.__dict__)
        self.solve()

    def search_args(self):
        if self.args.arg_mode == 'grid':
            return list(ParameterGrid(self.model_args_grid))
        elif self.args.arg_mode == 'fix':
            ret = {}
            d = self.model_args_grid
            for k in d:
                ret[k] = d[k][0]
            return [ret]
        else:
            args = []
            ret = {}
            d = self.model_args_grid
            for k in d:
                ret[k] = d[k][0]
            for k in sorted(d.keys()):
                if len(d[k]) > 1:
                    for v in d[k][1:]:
                        ret[k] = v
                        args.append(dict(ret))
                    ret[k] = d[k][0]
            if len(args) == 0 or self.args.arg_mode == '0':
                args = [ret] + args
            return args

    def get_args(self, Model):
        if not Model.add_global_args:
            return Model.tc
        grid_fix = False

        grid = {}
        # grid['rank'] = [False]
        grid['rank'] = [True]
        dataset.rank_metric = grid['rank']

        grid['add_features'] = [False]

        grid['batch_size'] = [32]
        if grid['rank'][0]:
            grid['batch_steps'] = [10000]
        else:
            grid['batch_steps'] = [3000]

        grid['dim_k'] = [64]
        # grid['init_word'] = ['word2vec', 'random']
        grid['init_word'] = ['word2vec']
        # grid['train_word'] = ['fix', 'train', 'dense']
        grid['train_word'] = ['dense']

        # sample, pos-neg, all, question
        # grid['pair_mode'] = ['sample', 'pos-neg', 'all']
        grid['pair_mode'] = ['sample']

        # grid['init_user'] = ['word2vec', 'random']
        grid['init_user'] = ['random']
        # grid['train_user'] = ['fix', 'train', 'dense']
        grid['train_user'] = ['train']

        grid['ans_bias'] = [True, False]
        grid['user_bias'] = [True, False]

        grid['opt'] = ['adam']
        # grid['lr'] = [1e-5]
        grid['lr'] = [1e-5]
        # grid['lr_emb'] = [1e-3, 1e-4, 1e-5]
        grid['lr2'] = [None]
        # grid['mc'] = [0.1, 0.3, 0.9]
        grid['mc'] = [1]

        grid['init_f'] = ['uniform']  # normal, uniform
        # grid['init_f'] = ['normal'] # normal, uniform
        grid['init_v'] = [0.05]  # 0.05 for uniform, 0.01 for normal

        grid['fc'] = [[512]]
        grid['text_emb_relu'] = [False]
        grid['dp'] = [0]

        if grid_fix:
            for k in grid.keys():
                grid[k] = grid[k][:1]

        grid.update(Model.tc)
        return grid
