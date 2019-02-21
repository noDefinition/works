import pandas as pd
from collections import OrderedDict as Od

from clu.data.datasets import *
import utils.array_utils as au
import utils.io_utils as iu


def analyze_refine_mean_and_stderr(result_file, mean_std_file):
    using_scores = ['nmi', 'homo', 'cmplt', 'ari']
    arg_tpc_clu_list = iu.load_array(result_file)
    rows = list()
    for kwargs, topics, clusters in arg_tpc_clu_list:
        scores = [au.score(topics, clusters, s) for s in using_scores]
        res_dict = Od(zip(using_scores, scores))
        row = Od(list(kwargs.items()) + list(res_dict.items()))
        rows.append(row)
    rows = sorted(rows, key=lambda item: item['nmi'], reverse=True)
    df = pd.DataFrame(data=rows)
    print(df)
    score_array = df[using_scores].values
    mean = np.mean(score_array, axis=0)
    std = np.std(score_array, axis=0, ddof=1)
    table = list(zip(*[using_scores, mean, std]))
    lines = ['{}: {} ± {}'.format(name, round(mean, 4), round(std, 4)) for name, mean, std in table]
    iu.write_lines(mean_std_file, lines)


def analyze_glda_result():
    glda_base = '/home/cdong/works/research/clu/baselinee/GaussianLDA'
    out_dir_ = iu.join(glda_base, 'output_{}/')
    d_classes = (DataReuters(), Data20ng())
    
    # for d in d_classes:
    #     # topic_list = d.get_topic_list()
    #     out_dir = out_dir_.format(d.name)
    # # for out_dir in fi.listchildren(glda_base, children_type=fi.TYPE_DIR, pattern='output'):
    # #     print(out_dir)
    #     for param in fi.listchildren(out_dir, children_type=fi.TYPE_DIR, concat=True):
    #         print(param[param.rfind('/') + 1:])
    #         for assignment in fi.listchildren(param, concat=True, pattern='table_assignments_'):
    #             lines = fu.read_lines(assignment)
    #             print(len(lines))
    from collections import Counter
    import utils.array_utils as au
    import re
    
    def fn(f):
        return f[f.rfind('/') + 1:]
    
    for out_dir in iu.list_children(glda_base, ctype=iu.DIR, pattern='^output', full_path=True):
        # topic_list = Data20ng().get_topic_list()
        dname = re.findall('output_(.+)$', out_dir)[0]
        print(dname)
        topic_list = name2object[dname].get_topics()
        
        print(out_dir)
        for param in iu.list_children(out_dir, ctype=iu.DIR, full_path=True):
            print(param[param.rfind('/') + 1:])
            for assign in iu.list_children(param, pattern='table_assignments', full_path=True):
                lines = iu.read_lines(assign)
                if len(lines) != len(topic_list):
                    continue
                print(fn(assign), len(lines))
                print(lines[100])
                cluster_list = [Counter(list(map(int, line.split_length()))).most_common()[0][0] for line in lines]
                print(au.score(topic_list, cluster_list, 'nmi'))
                print(au.score(topic_list, cluster_list, 'ari'))


def analyze_glda_result2():
    glda_base = '/home/cdong/works/research/clu/baselinee/GaussianLDA'
    out_dir_ = iu.join(glda_base, 'output_{}/')
    d_classes = (DataReuters(), Data20ng())
    
    # for d in d_classes:
    #     # topic_list = d.get_topic_list()
    #     out_dir = out_dir_.format(d.name)
    # # for out_dir in fi.listchildren(glda_base, children_type=fi.TYPE_DIR, pattern='output'):
    # #     print(out_dir)
    #     for param in fi.listchildren(out_dir, children_type=fi.TYPE_DIR, concat=True):
    #         print(param[param.rfind('/') + 1:])
    #         for assignment in fi.listchildren(param, concat=True, pattern='table_assignments_'):
    #             lines = fu.read_lines(assignment)
    #             print(len(lines))
    import utils.array_utils as au
    import re
    
    def fn(f):
        return f[f.rfind('/') + 1:]
    
    for out_dir in iu.list_children(glda_base, ctype=iu.DIR, pattern='^output', full_path=True):
        # topic_list = Data20ng().get_topic_list()
        dname = re.findall('output_(.+)$', out_dir)[0]
        print('fuck', dname)
        topic_list = name2object[dname].get_topics()
        for param in iu.list_children(out_dir, ctype=iu.DIR, full_path=True):
            print(param[param.rfind('/') + 1:])
            for assign in iu.list_children(param, pattern='document_topic', full_path=True):
                lines = iu.read_lines(assign)
                if len(lines) != len(topic_list):
                    continue
                # print(fn(assign), len(lines))
                # print(lines[100])
                # cluster_list = [Counter(list(map(int, line.split()))).most_common()[0][0] for line in lines]
                clu_assign = [list(map(float, line.split_length())) for line in lines]
                cluster_list = np.argmax(clu_assign, axis=1)
                print(au.score(topic_list, cluster_list, 'nmi'), au.score(topic_list, cluster_list, 'ari'))


if __name__ == '__main__':
    # _d_class = DataTREC
    # _topic_clu_file = 'GSDMM_{}_topic_clu.txt'.format(_d_class.name)
    # _mean_std_file = 'GSDMM_{}_mean_std.txt'.format(_d_class.name)
    #
    # analyze_refine_mean_and_stderr(_topic_clu_file, _mean_std_file)
    # analyze_glda_result()
    analyze_glda_result2()
