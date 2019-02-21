import os
import pickle
import sys


def get_features(text):
    tlen = len(text)
    unique_word = len(set(text))
    unique_word_rate = unique_word / tlen
    return [tlen, unique_word, unique_word_rate]


Home = '/home/wwang/Projects/QAR_data'


def main():
    print('hello world, ans_text_features.py')
    ds = 'test'
    # ds = 'so'
    if len(sys.argv) > 1:
        ds = sys.argv[-1]
    home = '{}/{}_data'.format(Home, ds)
    to_dir = 'features/ans_text'
    os.system('mkdir {}/{}'.format(home, to_dir))
    
    # aid2text = pickle.load(open('{}/raw/aid2text_dict.pkl'.format(home), 'rb'))
    aid2wids = pickle.load(open('{}/features/aid2wids_dict.pkl'.format(home), 'rb'))
    aid2features = {}
    for aid in aid2wids:
        aid2features[aid] = get_features(aid2wids[aid])
    pickle.dump(aid2features, open('{}/{}/aid2features_dict.pkl'.format(home, to_dir), 'wb'),
                protocol=4)


if __name__ == '__main__':
    main()
