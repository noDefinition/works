#########################################################################
# File Name: sklean_models.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2018年08月21日 星期二 14时25分09秒
#########################################################################

allf = 'auqmf'
#allf = 'uf'
init = ''
#allf = [[i] for i in range(7)]
#init = []
def get_all_f():
    m = len(allf)
    for i in range(0, 1 << m):
        r = init
        for j in range(m):
            if ((i >> j) & 1):
                r = r + allf[j]
        if r:
            yield r
#print(list(get_all_f()))

import numpy as np, sys, math, os
from . import utils
from . import dataset
import time
from . import log
from . import q_maxlen, a_maxlen
from . import BasicPair

class Ridge(BasicPair):
    Data = dataset.sk_data
    tc = {}
    tc['alpha'] = [0.3, 1, 3, 10, 30][::-1]
    #tc['alpha'] = [0.1, 0.3] + list(range(30))
    #tc['alpha'] = [1, 3]
    #tc['using_data'] = ['a', 'u', 'au', 'aq', 'uq', 'auq', 'am', 'um', 'aum', 'auqm']
    #tc['using_data'] = ['aq', 'uq', 'auq', 'am', 'um', 'aum', 'auqm']
    #tc['using_data'] = ['auf', 'au', 'aumf', 'aum', 'auqmf', 'auqm']
    tc['using_data'] = ['auq']
    #tc['using_data'] = list(get_all_f())
    #tc['using_features'] = list(get_all_f())
    tc['using_features'] = [list(range(7))]
    #tc['using_data'] = ['u']
    tc['normalize'] = [True, False]
    add_global_args = False

    def init(self):
        self.data = self.Data(self.dataset)

    def fit(self, args, args_i, args_n):
        self.args_i = args_i
        self.args_n = args_n
        self.args = args
        self.start_time = time.time() - 5 * 60
        from sklearn.linear_model import Ridge as Model
        model = Model(alpha = args.get('alpha', 1.0), normalize = args['normalize'])
        xy, info = self.data.get_xy_info('train', args['using_data'], args['using_features'])
        utils.timer.start()
        model.fit(*xy)
        train_time = utils.timer.stop()
        utils.timer.start()
        vali = self.data.evaluate('vali', model.predict, args['using_data'], args['using_features'])
        vali_time = utils.timer.stop()
        msg = 'vali: {}, time: {:.1f}s {:.1f}s'.format(vali, train_time, vali_time)
        log(msg, i = -1, red = True)
        return self.data.evaluate('test', model.predict, args['using_data'], args['using_features'])


def main():
    print('hello world, sklean_models.py')

if __name__ == '__main__':
    main()

