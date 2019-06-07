import pandas as pd
from utils import iu, au
from cqa.mee import K
from cqa.mee.evaluate import MeanRankScores

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


class AnalyzeMee:
    def __init__(self):
        from argparse import ArgumentParser
        self.parser = ArgumentParser()
        self.parser.add_argument('-c', action='store_true', default=False, help='if choose path')
        self.parser.add_argument('-s', action='store_true', default=False, help='if summary to csv')
        self.parser.add_argument('-d', action='store_true', default=False, help='if show detail')
        self.exclude = {K.lid, K.gi, K.gp, K.es, K.ep, }
        self.args = self.parser.parse_args()

    def should_summary(self):
        return self.args.s

    def get_log_path(self):
        cand_paths = iu.list_children('./', iu.DIR, r'^log\d', full_path=True)
        # if len(cand_paths) == 0:
        cand_paths += iu.list_children('./logs', iu.DIR, r'^log\d', full_path=True)
        log_path = iu.choose_from(cand_paths) if self.args.c else iu.most_recent(cand_paths)
        return log_path

    def main(self):
        log_path = self.get_log_path()
        print('log path:', log_path)
        log_files = iu.list_children(log_path, pattern=r'^gid.+\.txt$', full_path=True)
        best_list = list()
        pre = 't_'
        pre = ''
        print('pre', pre)
        for file in log_files:
            entries = au.name2entries(name=iu.get_name(file), postfix='.txt', exclude=self.exclude)
            scores = [iu.loads(l) for l in iu.read_lines(file) if
                      (l.startswith('{') and 'NDCG' in l)]
            scores_with_test = [s for s in scores if '%sNDCG' % pre in s]
            # scores_with_test = [s for s in scores if 'NDCG' in s]
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
        # pre = 't_'
        table[temp] = table['%sNDCG' % pre] + table['%sMAP' % pre] + table['%sMRR' % pre]
        table = table.sort_values(by=temp)
        # table.drop([temp, K.lr, K.reg, K.gid, K.ep], axis=1, inplace=True)
        if self.args.s:
            table.to_csv(iu.join(log_path, 'summary.csv'))

        # group_col = [K.dn, K.atp, K.vs]
        group_col = [K.dn, K.woru, K.topk]
        res = pd.DataFrame()
        idx = 0
        for value, df in table.groupby(group_col):
            dic = dict(zip(group_col, value))
            mean = df.drop(group_col, axis=1).mean()
            dic.update(mean.to_dict())
            for k, v in dic.items():
                res.loc[idx, k] = v
            idx += 1
        print(res)
        res.to_csv('mean.csv')
        return

        # group_col = [K.dn]
        # grouped = table.groupby(group_col)
        # kv_df_list = list()
        # summ = pd.DataFrame()
        # import numpy as np
        # for idx, (values, table) in enumerate(grouped):
        #     # print(list(zip(group_col, values)))
        #     kv = dict(zip(group_col, values))
        #     kv['final'] = np.mean(table['v_NDCG'] + table['v_MAP'] + table['v_MRR']) / 3
        #     kv['final'] = kv['final'].round(3)
        #     kv_df_list.append([kv, table])
        #     columns = ['%s_%s' % (a, b) for a in ['v', 't'] for b in ['NDCG', 'MAP', 'MRR']]
        #     s = table[columns].mean(0)
        #     print(dict(s))
        #     # print(s.index)
        #     # print(s[s.index])
        #     # print(list(s.data))
        #     # summ.loc[idx, 'data'] = values
        #     # summ.loc[idx, columns] = list(s.data)
        #     summ.append(dict(s), ignore_index=True)
        #     # print(table, '\n')
        # print(summ)

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

    def full_score_ver(self):
        s_names = MeanRankScores.s_names
        k_values = MeanRankScores.k_values

        log_path = self.get_log_path()
        print('log path:', log_path)
        best_list = list()
        pre = 't_'
        # log_path = '/home/cdong/works/cqa/mee/base/d2v'
        # pre = ''
        wanted_scores = ['{}{}@{}'.format(pre, s, k) for s in s_names for k in k_values[1:3]]
        for file in iu.list_children(log_path, pattern=r'^gid.+\.txt$', full_path=True):
            entries = au.name2entries(name=iu.get_name(file), postfix='.txt', exclude=self.exclude)
            test_scores = [iu.loads(l.replace('\'', '"')) for l in iu.read_lines(file) if
                           l.startswith('{') and '{}NDCG'.format(pre) in l]
            # test_scores = [d for d in scores if 't_NDCG' in d]
            if len(test_scores) == 0:
                print(au.entries2name(entries), 'lacks test info')
                continue
            score_table = pd.DataFrame()
            best_tests = test_scores[-3:]
            for idx, rvs2scores in enumerate(best_tests):
                for title, value in rvs2scores.items():
                    score_table.loc[idx, title] = value
            score_table = score_table.mean(axis=0).round(4)
            score_table = score_table[wanted_scores]
            best_list.append((dict(entries), score_table.to_dict()))

        table = pd.DataFrame()
        for i, (name2param, sname2score) in enumerate(best_list):
            for k, v in list(name2param.items()) + list(sname2score.items()):
                table.loc[i, k] = v
        table.fillna('-', inplace=True)
        print(table)

        temp = 'mmm'
        table[temp] = 0
        for s in s_names:
            for k in k_values[1:3]:
                table[temp] += table['{}{}@{}'.format(pre, s, k)]
        table.sort_values(by=temp, inplace=True)

        log_name = iu.get_name(log_path)
        drop_col = {temp, K.lr, K.reg, K.gid, K.ep, K.fda, K.temp}
        group_col = [K.dn, K.vs]
        if 'lstm_wwoatt' in log_name:
            group_col += [K.atp, K.topk]
        elif 'tune_topk' in log_name or 'other_topk' in log_name:
            group_col += [K.woru, K.topk]
            drop_col.add(K.drp)
        elif 'only_user' in log_name:
            group_col += [K.woru]
            drop_col.update({K.topk, K.drp})
        elif 'qtypes' in log_name or 're_so' in log_name or 'qau_cat' in log_name:
            group_col += [K.qtp]
        elif 'margin_drop' in log_name:
            group_col += [K.woru, K.drp, K.topk]
        else:
            # group_col = [K.gid]
            raise ValueError('cdong: dont know how to organize hyperparams:', log_name)
        table.drop(list(drop_col), axis=1, inplace=True, errors='ignore')

        res = pd.DataFrame()
        idx = 0
        for value, df in table.groupby(group_col):
            dic = dict(zip(group_col, value))
            mean = df.drop(group_col, axis=1).mean()
            dic.update(mean.to_dict())
            for k, v in dic.items():
                res.loc[idx, k] = v
            idx += 1
        print(res)
        res.to_csv('{}.csv'.format(iu.get_name(log_path)))
        if self.args.s:
            table.to_csv(iu.join(log_path, '{}_summary.csv'.format(iu.get_name(log_path))))
        return


if __name__ == '__main__':
    AnalyzeMee().full_score_ver()
