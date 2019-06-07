from argparse import ArgumentParser


class Args:
    @staticmethod
    def get_parser() -> ArgumentParser:
        def _(name, *args, **kwargs):
            prev('-' + name, *args, **kwargs)

        parser = ArgumentParser()
        prev = parser.add_argument
        parser.add_argument = _
        return parser

    def parse_args(self):
        return dict(self.parser.parse_args().__dict__)

    def add_arg(self, name, *args, **kwargs):
        self.parser.add_argument(name, *args, **kwargs)
        return name

    def __init__(self):
        self.parser: ArgumentParser = self.get_parser()
        # 通用的参数
        self.gid = self.add_arg('gid', type=int, help='global id', required=True)
        self.lid = self.add_arg('lid', type=int, help='local id (同一组超参的重复次数)')
        self.gi = self.add_arg('gi', type=int, help='gpu id number')
        self.gp = self.add_arg('gp', type=float, help='gpu fraction')

        self.vs = self.add_arg('vs', type=str, help='model version')
        self.dn = self.add_arg('dn', type=str, help='data name')
        self.ep = self.add_arg('ep', type=int, help='epoch num')
        self.lg = self.add_arg('lg', type=str, help='logging path')


X = Args()
