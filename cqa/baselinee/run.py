import argparse
from pprint import pformat

from sklearn.model_selection import ParameterGrid

from prev import dataset, models
from prev.utils import log_dir, logger


# noinspection PyAttributeOutsideInit
class Run:
    def __init__(self):
        self.args = self.parse_args()
        self.run()

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='models_prev')
        _ = parser.add_argument
        _('-model', type=str, default='BasicModel')
        _('-ds', '--dataset', type=str, default='test')
        # self.add_arg('-rt', '--run_times', type = int, default = 1)
        _('-am', '--arg_mode', type=str, default='fix')
        _('-msg', '--message', type=str, default='')

        _('-summ', '--run_summary', type=int, default=1)
        _('-test', '--run_test', type=int, default=0)

        _('-es', '--early_stop', type=int, default=5)
        _('-mee', '--max_epochs', type=int, default=100)
        _('-gpu', type=str, default='3')
        return parser.parse_args()

    def set_logfn(self):
        if self.args.massage:
            logfn = '{} {}'.format('fuck_you', self.args.message)
        else:
            logfn = '{} {}'.format(self.args.model, self.args.dataset)
        logger.set_fn(home=log_dir, fn=logfn)

    def log_msg(self):
        logger.log('solve model: %s' % self.args.model, red=True)
        logger.log('dataset: {}'.format(self.args.dataset), red=True)
        logger.log(pformat(self.grid))

    def summary(self):
        if len(self.history) > 1:
            logger.log('\n\n{} summary begin!'.format(self.args.model), red=True)
            for h in self.history:
                logger.log(h)
            logger.log('best performance:', red=True)
            logger.log(self.best_msg)
            logger.log('summary end!\n\n', red=True)
        logger.log('log file: {}\nnow: {}'.format(logger.fn, logger.date()))
        print('\n\n')

    def solve(self):
        best_result = None
        self.history = list()
        self.best_msg = 'None'
        search_args = list(self.search_args())
        n = len(search_args)
        for i, model_args in enumerate(search_args):
            i += 1
            prt_args = self.prt_args(model_args)
            logger.log('\n#args: {}/{} {}'.format(i, n, prt_args), red=True)
            result = self.model.fit(model_args, i, n)
            msg = 'result: {}, args: {}'.format(result, prt_args)
            logger.log(msg, red=True)
            self.history.append(msg)
            if result.is_better_than(best_result):
                best_result = result
                self.best_msg = msg
        logger.log(best_result.prt())
        self.summary()

    def prt_args(self, model_args):
        ret = {}
        for k in model_args:
            if len(self.grid[k]) > 1:
                ret[k] = model_args[k]
        return ret

    def run(self):
        self.grid = self.get_grid()
        self.set_logfn()
        self.log_msg()
        self.model = vars(models)[self.args.model](**self.args.__dict__)
        self.solve()

    def search_args(self):
        if self.args.arg_mode == 'grid':
            return list(ParameterGrid(self.grid))
        elif self.args.arg_mode == 'fix':
            ret = {}
            d = self.grid
            for k in d:
                ret[k] = d[k][0]
            return [ret]
        else:
            args = []
            ret = {}
            d = self.grid
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

    def get_grid(self):
        grid = dict()
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

        # grid['pair_mode'] = ['sample', 'pos-neg', 'all']
        grid['pair_mode'] = ['sample']

        # grid['init_user'] = ['word2vec', 'random']
        grid['init_user'] = ['random']
        # grid['train_user'] = ['fix', 'train', 'dense']
        grid['train_user'] = ['train']

        grid['ans_bias'] = [True, False]
        grid['user_bias'] = [True, False]

        grid['opt'] = ['adam']
        grid['lr'] = [1e-5]
        # grid['lr_emb'] = [1e-3, 1e-4, 1e-5]
        grid['lr2'] = [None]
        # grid['mc'] = [0.1, 0.3, 0.9]
        grid['mc'] = [1]

        grid['init_f'] = ['uniform', 'normal']
        grid['init_v'] = [0.05]  # 0.05 for uniform, 0.01 for normal

        grid['fc'] = [[512]]
        grid['text_emb_relu'] = [False]
        grid['dp'] = [0]

        grid['model'] = [BaseException]

        grid_fix = False
        if grid_fix:
            for k in grid.keys():
                grid[k] = grid[k][:1]
        # grid.update(Model.tc)
        return grid
