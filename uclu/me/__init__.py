from utils.tune.arg_keys import Args


class UcluArgs(Args):
    def __init__(self):
        super(UcluArgs, self).__init__()

        self.cn = self.add_arg('cn', type=int, help='cluster num')
        self.lr = self.add_arg('lr', type=float, help='learning rate')
        self.tpad = self.add_arg('tpad', type=int, help='padding of title')
        self.bpad = self.add_arg('bpad', type=int, help='padding of body')

        self.bs = self.add_arg('bs', type=int, help='pos batch size')
        self.ns = self.add_arg('ns', type=int, help='neg batch num')
        self.ed = self.add_arg('ed', type=int, help='embed dim')
        self.hd = self.add_arg('hd', type=int, help='hidden dim')
        self.md = self.add_arg('md', type=int, help='model inner dim')


U = UcluArgs()
