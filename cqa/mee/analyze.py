import pandas as pd
from utils import iu, au
from cqa.mee import K

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


class AnalyzeMee:
    def __init__(self):
        from argparse import ArgumentParser
        self.parser = ArgumentParser()
        self.parser.add_argument('-c', action='store_true', default=False, help='if choose path')
        self.parser.add_argument('-s', action='store_true', default=False, help='if summary to csv')
        self.parser.add_argument('-d', action='store_true', default=False, help='if show detail')
        self.exclude = {K.gi, K.gp, K.es, K.ep, K.sc, K.lid, K.fda, K.bs, K.drp}
        self.args = self.parser.parse_args()

    def should_summary(self):
        return self.args.s

    def get_log_path(self):
        cand_paths = iu.list_children('./', iu.DIR, r'^log\d', full_path=True)
        if len(cand_paths) == 0:
            cand_paths = iu.list_children('./logs', iu.DIR, r'^log\d', full_path=True)
        log_path = iu.choose_from(cand_paths) if self.args.c else iu.most_recent(cand_paths)
        return log_path

    def main(self):
        log_path = self.get_log_path()
        print('log path:', log_path)
        log_files = iu.list_children(log_path, pattern=r'^gid.+\.txt$', full_path=True)
        best_list = list()
        for file in log_files:
            entries = au.name2entries(name=iu.get_name(file), postfix='.txt', exclude=self.exclude)
            scores = [iu.loads(l) for l in iu.read_lines(file) if
                      (l.startswith('{') and 'v_NDCG' in l)]
            scores_with_test = [s for s in scores if 't_NDCG' in s]
            if len(scores) == 0 or len(scores_with_test) == 0:
                print(au.entries2name(entries), 'lacks test info')
                continue
            best_scores = scores_with_test[-3:]
            name2score = pd.DataFrame()
            for idx, rvs2scores in enumerate(best_scores):
                rvs2scores.pop('brk_cnt')
                for title, value in rvs2scores.items():
                    name2score.loc[idx, title] = value
                # for rvs, score in rvs2scores.items():
                #     for name, value in score.items():
                #         title = '{}_{}'.format(rvs[0], name)
            name2score = name2score.mean(axis=0).round(4)
            name2score['ep'] = len(scores)
            best_list.append((dict(entries), name2score.to_dict()))

        table = pd.DataFrame()
        for i, (name2param, name2score) in enumerate(best_list):
            for k, v in list(name2param.items()) + list(name2score.items()):
                table.loc[i, k] = v
        table.fillna('-', inplace=True)
        temp = 'mmm'
        pre = 't'
        table[temp] = table['%s_NDCG' % pre] + table['%s_MAP' % pre] + table['%s_MRR' % pre]
        table = table.sort_values(by=temp)
        table.drop([temp, K.lr, K.reg], axis=1, inplace=True)
        # table = table.query('dpt=="1"')
        if self.args.s:
            table.to_csv(iu.join(log_path, 'summary.csv'))

        # print(table.columns)
        # print(table)
        # group_col = [K.dn, K.mix, K.act, K.dpt]

        for value, df in table.groupby(K.vs):
            df.pop(K.ep)
            print(value)
            print(df)
            mean = df.groupby(K.dn).mean()
            print(mean)
            mean.to_csv('%s.csv' % value)
        return

        group_col = [K.dn]
        grouped = table.groupby(group_col)
        kv_df_list = list()
        summ = pd.DataFrame()
        import numpy as np
        for idx, (values, table) in enumerate(grouped):
            # print(list(zip(group_col, values)))
            kv = dict(zip(group_col, values))
            kv['final'] = np.mean(table['v_NDCG'] + table['v_MAP'] + table['v_MRR']) / 3
            kv['final'] = kv['final'].round(3)
            kv_df_list.append([kv, table])
            columns = ['%s_%s' % (a, b) for a in ['v', 't'] for b in ['NDCG', 'MAP', 'MRR']]
            s = table[columns].mean(0)
            print(dict(s))
            # print(s.index)
            # print(s[s.index])
            # print(list(s.data))
            # summ.loc[idx, 'data'] = values
            # summ.loc[idx, columns] = list(s.data)
            summ.append(dict(s), ignore_index=True)
            # print(table, '\n')
        print(summ)


        # sum_df = pd.DataFrame()
        # kv_df_list = sorted(kv_df_list, key=lambda x: x[0]['final'])
        # for kv, df in kv_df_list:
        #     if kv[K.dn] == 'zh':
        #         print(len(df), au.entries2name(kv, inter=', '))
        #         sum_df = sum_df.append(kv, ignore_index=True)
        #     if self.args.d:
        #         print(df, '\n')
        # print('\n')
        # for kv, df in kv_df_list:
        #     if kv[K.dn] == 'so':
        #         print(len(df), au.entries2name(kv, inter=', '))
        #         sum_df = sum_df.append(kv, ignore_index=True)
        #     if self.args.d:
        #         print(df, '\n')
        # print
        # sum_df.to_csv('sum.csv')


if __name__ == '__main__':
    AnalyzeMee().main()
