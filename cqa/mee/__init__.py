from utils.tune.arg_keys import Args


class CqaArgs(Args):
    def __init__(self):
        super(CqaArgs, self).__init__()

        self.fda = self.add_arg('fda', type=int, help='0=simple data, 1=full', required=True)
        self.es = self.add_arg('es', type=int, help='runtime: step before early stop')
        self.lr = self.add_arg('lr', type=float, help='train: learning rate')

        self.reg = self.add_arg('reg', type=float, default=0, help='coeff of regularization')
        self.drp = self.add_arg('drp', type=float, default=1, help='probability of dropout')
        self.temp = self.add_arg('temp', type=float, default=0, help='temperature in kl/softmax')

        self.woru = self.add_arg('woru', type=int, help='adv noise for word/user, 0=no adv')
        self.topk = self.add_arg('topk', type=int, help='topk for mean pooling')
        self.eps = self.add_arg('eps', type=float, help='epsilon for adv gradient')
        self.atp = self.add_arg('atp', type=int, help='adv type')
        self.qtp = self.add_arg('qtp', type=int, help='question type')
        self.ttp = self.add_arg('ttp', type=int, help='text type')

        self.mix = self.add_arg('mix', type=float, help='mixture weight of extra modules')
        self.dpt = self.add_arg('dpt', type=int, help='depth of dense')
        self.act = self.add_arg('act', type=int, help='activate function of dense')


K = CqaArgs()
