from cqa.mee.analyze import AnalyzeMee
from cqa.baselinee import J


class AnalyzeBase(AnalyzeMee):
    def __init__(self):
        super(AnalyzeBase, self).__init__()
        self.exclude = {J.gi, J.gp, J.es, J.ep, J.sc, J.lid, J.fda, J.bs, J.drp}


if __name__ == '__main__':
    AnalyzeBase().main()
