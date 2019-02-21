from subprocess import Popen, DEVNULL as V

import utils.multiprocess_utils as mu
import utils.timer_utils as tmu
from clu.data.datasets import *

from utils.node_utils import Nodes

d_class_ = Nodes.select(ngpu=Data20ng(), ncpu=Data20ng())
num_iter = 10
# data_name, embed_name = mfb.DataTREC.name, mfb.embed_glovec_27B
# standard_base, docarr_file, embed_file, _ = mfb.get_standard_files(mfb.base_path_, data_name, embed_name)
# embed_str_file_, corpus_file_ = mfb.get_method_files(standard_base, mfb.method_glda)
# dimension = {
#     mfb.embed_glovec_27B: 200, mfb.embed_glovec_6B: 300, mfb.embed_google_news: 300
# }[mfb.embed_google_news]
dimension = 300
embed_file_, corpus_file_ = d_class_.glda_embed_file, d_class_.glda_corpus_file
K = d_class_.topic_num_list

# embed_str_file_ = '/home/nfs/cdong/research/clu/baselinee/GaussianLDA/data/nips/embed.new'
# corpus_file_ = '/home/nfs/cdong/research/clu/baselinee/GaussianLDA/data/nips/corpus.new'
# dimension = 300
# K = 50

glda_base_ = '/home/cdong/works/research/clu/baselinee/GaussianLDA'
bin_dir_ = iu.join(glda_base_, 'bin/')
# out_dir_ = fi.join(glda_base_, 'output_{}/'.format(d_class_.name))
out_dir_ = iu.join(glda_base_, 'output_{}/')
external_ = ":".join(["external_libs/ejml-0.25.jar",
                      "external_libs/commons-logging-1.2/commons-logging-1.2.jar",
                      "external_libs/commons-math3-3.3/commons-math3-3.3.jar"])
complie_ = 'javac -sourcepath src/ -d bin/ -cp "{}" src/sampler/GaussianLDA.java '.format(external_)
execute_ = 'java  -Xmx70g -cp "bin/:{}" sampler/GaussianLDA '.format(external_)


def compile_glda():
    iu.mkdir(bin_dir_, rm_prev=True)
    p = Popen(complie_, cwd=glda_base_, shell=True)
    p.communicate()


def run_glda(command, cwd):
    # p = Nodes.select(ncpu=False, ngpu=True)
    # if p:
    #     Popen(command, cwd=cwd, shell=True, bufsize=1).communicate()
    # else:
    Popen(command, cwd=cwd, shell=True, bufsize=1, stdin=V, stdout=V, stderr=V).communicate()


def run_glda_multi():
    d_classes = (DataReuters(), Data20ng())
    for d in d_classes:
        iu.mkdir(out_dir_.format(d.name), rm_prev=True)
    # file_name_list = [embed_file_, corpus_file_, out_dir_]
    # name_value_list = [
    #     ('D', [300]),
    #     ('K', [250]),
    #     ('K_0', [0.1, 0.01, 0.001]),
    #     ('alpha', [0.0001, 0.004, 0.01]),
    #     ('numIterations', [5]),
    #     ('round_index', [666]),
    # ]
    name_value_list = [(
        ('f1', [d.glda_embed_file]),
        ('f2', [d.glda_corpus_file]),
        ('f3', [out_dir_.format(d.name)]),
        ('D', [dimension]),
        ('K', [d.topic_num]),
        ('numIterations', [num_iter]),
        ('round_index', [i for i in range(1)]),
        ('alpha', [0]),
        ('k_0', [1.0, 0.1, 0.01, 0.001]),
    ) for d in d_classes]
    # file_args_str = ('{} ' * len(file_name_list)).format(*file_name_list)
    # file_args_str = ' '.join(file_name_list)
    # value_args_str = '{} ' * len(name_value_list)
    # args_list = [(execute_ + file_args_str + value_args_str.format(*g.values()), glda_base_) for g in grid]
    grid = au.merge(au.grid_params(nv) for nv in name_value_list)
    args_list = [(execute_ + ' '.join(list(map(str, g.values()))), glda_base_) for g in grid]
    print(args_list[:4])
    mu.multi_process_batch(run_glda, batch_size=Nodes.select(ncpu=32, ngpu=4), args_list=args_list)


def analyze_one_cluster(out_dir, d_file, e_file, c_file):
    def lines_mapper(lines, func):
        return [list(map(func, line.strip().split_length())) for line in lines]
    
    print(out_dir)
    # clu_files = fi.listchildren(out_dir, pattern='^\d', concat=True)
    # print('num of cluster files:', len(clu_files))
    doc_clusters_list = iu.read_lines(iu.join(out_dir, "table_assignments.txt"))
    doc_clusters_list = lines_mapper(iu.read_lines(doc_clusters_list), int)
    doc_tokenids_list = lines_mapper(iu.read_lines(c_file), int)
    print('read over')
    exit()
    doc_clusters = au.merge(doc_clusters_list)
    doc_tokenids = au.merge(doc_tokenids_list)
    max_cluid = max(doc_clusters)
    assert K == max_cluid
    print(max_cluid)
    cluster_tokenids_list = [[] for _ in range(max_cluid)]
    assert len(doc_clusters) == len(doc_tokenids)
    for i in range(len(doc_clusters)):
        cluster, tokenid = doc_clusters[i], doc_tokenids[i]
        cluster_tokenids_list[cluster].append(tokenid)
    # for i in range(len(doc_tokenids_list)):
    #     tokenids, clusters = doc_tokenids_list[i], doc_clusters_list[i]
    #     assert len(tokenids) == len(clusters)
    #     for j in range(len(tokenids)):
    #         tokenid, cluster = tokenids[j], clusters[j]
    #         cluster2tokenids[cluster].append(tokenid)
    # print('len stats:', [len(tokenids) for tokenids in cluster_tokenids_list])
    
    embed_table = np.load(e_file)
    cluster_centers = [np.mean(embed_table[tokenids], axis=0) for tokenids in cluster_tokenids_list]
    document_centers = [np.mean(embed_table[tokenids], axis=0) for tokenids in doc_tokenids_list]
    cos_sim = au.cosine_similarity(document_centers, cluster_centers)
    print('shape of one vector is:', cluster_centers[10].shape)
    print('cos_sim shape is:', cos_sim.shape)
    clusters_list = np.argmax(cos_sim, axis=1)
    topics_list = [d.topic for d in du.load_docarr(d_file)]
    nmi = au.score(topics_list, clusters_list, 'nmi')
    print('nmi:', nmi)


def analyze_glda_result():
    d_classes = (DataReuters(), Data20ng())
    for d in d_classes:
        topic_list = d.get_topics()
        out_dir = out_dir_.format(d.name)
        for param in iu.list_children(out_dir, full_path=True):
            print(param[param.rfind('/') + 1:])
            for assignment in iu.list_children(param, full_path=True, pattern='table_assignments_'):
                lines = iu.read_lines(assignment)
                print(len(lines))
    # # out_dir_ = "/home/nfs/cdong/research/clu/baselinee/GaussianLDA/output_change_embedding/"
    # res_dirs = fi.listchildren(out_dir_, fi.TYPE_DIR, concat=True)[:1]
    # for res_dir in res_dirs:
    #     analyze_one_cluster(res_dir, docarr_file, embed_file, corpus_file_)


# def make_embedding_for_glda_data():
#     embed_file = "glovec6B/glove.6B.300d.txt"
#     embed_file = "glovecTwitter27B/glove.twitter.27B.200d.txt"
#     vocab_file = "/home/nfs/cdong/research/clu/baselinee/GaussianLDA/data/20_news/vocab.txt"
#     embed_lines = fu.read_lines(embed_file)
#     vocabulary = fu.read_lines(vocab_file)
#     word2vec = dict()
#     for line in embed_lines:
#         splits = line.strip().split(' ')
#         # word2vec[splits[0]] = list(map(float, splits[1:]))
#         word2vec[splits[0].lower()] = 0
#     print(set(vocabulary).difference(set(word2vec.keys())))


# def analyze_original_data():
#     vocab_file = "/home/nfs/cdong/research/clu/baselinee/GaussianLDA/data/20_news/vocab.txt"
#     v_num = len(fu.read_lines(vocab_file))
#     t1 = "/home/nfs/cdong/research/clu/baselinee/GaussianLDA/data/20_news/corpus.test"
#     t2 = "/home/nfs/cdong/research/clu/baselinee/GaussianLDA/data/20_news/corpus.train"
#     lines = fu.read_lines(t1) + fu.read_lines(t2)
#     t_num = len(set(au.merge_array([line.split(' ') for line in lines])))
#     print(v_num, t_num)


if __name__ == '__main__':
    tmu.check_time()
    # analyze_original_data()
    # make_embedding_for_glda_data()
    # tmu.check_time()
    
    compile_glda()
    tmu.check_time()
    run_glda_multi()
    tmu.check_time()
    # analyze_glda_result()
    # tmu.check_time()
