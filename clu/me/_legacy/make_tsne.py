import utils.io_utils as iu
import utils.reduction_utils as ru
from data.datasets import *


def result_files2with_cluster_tsne(res_files):
    def convert(a):
        return np.array([np.array(v) for v in a])
    
    embed_list, name_list = list(), list()
    for file in res_files:
        _, _, _, doc_embed, _, clu_embed, dname = np.load(file)
        doc_embed, clu_embed = convert(doc_embed), convert(clu_embed)
        print(type(doc_embed), type(clu_embed))
        # embed = np.concatenate([doc_embed, clu_embed], axis=0)
        embed = doc_embed
        embed_list.append(embed)
        name_list.append(dname)
        # f_both = './original_embeddings/tsne_{}_both.npy'.format(dname)
        # np.save(f_both, embed)
    print('embed shapes', [a.shape for a in embed_list])
    
    kw_arg = dict(early_exaggeration=30, n_iter=2000, n_iter_without_progress=100)
    tsne_list = ru.fit_multi(ru.fit_tsne, embed_list, [kw_arg] * len(embed_list))
    
    for dname, points in zip(name_list, tsne_list):
        d_class = name2object[dname]()
        topic_list = d_class.get_topics()
        print()
        f_doc = './original_embeddings/{}_our.txt'.format(dname)
        ru.points_to_file(points, topic_list, f_doc)
        
        continue
        f_clu = './original_embeddings/{}_clu.txt'.format(dname)
        topic_num = d_class.topic_num
        print(points[:-topic_num].shape)
        ru.points_to_file(points[:-topic_num], topic_list, f_doc)
        ru.points_to_file(points[-topic_num:], [0] * topic_num, f_clu)


def summarize_cluster_weight_distribution(res_files):
    for file in res_files:
        c = Counter()
        _, c_weight_list, _, doc_embed, _, clu_embed, dname = np.load(file)
        print(len(c_weight_list), len(doc_embed))
        for c_weight in c_weight_list:
            max_weight = np.max(c_weight)
            c[int(max_weight * 100)] += 1
        for i in sorted(c.keys()):
            print(i, round(c[i]/len(c_weight_list), 6))
        print('\n---\n')


if __name__ == '__main__':
    res_base = './logging_r2/'
    # res_files = [fi.listchildren(res_base, concat=True, pattern='^{}\.npy'.format(gid))[0] for gid in [2, 17]]
    res_files = [iu.list_children(res_base, full_path=True, pattern='^{}\.npy'.format(gid))[0] for gid in [69]]
    for r in res_files:
        print(r)
    result_files2with_cluster_tsne(res_files)
    # summarize_cluster_weight_distribution(res_files)
