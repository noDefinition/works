import tensorflow as tf

from bert.tensorflow import tokenization, modeling
from .my_model import MyBertModel


def build_model(bert_config_file, is_training=False):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model = MyBertModel(bert_config, is_training)
    return model


def load_model_params(init_checkpoint):
    tvars = tf.trainable_variables()
    assignment_map, initialized_var_names = \
        modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_var_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s | %s", var.name, var.shape, init_string)


def get_tokenizer(vocab_file, do_lower_case=True):
    return tokenization.FullTokenizer(vocab_file, do_lower_case)


def tokenize_texts(texts, seq_len, tokenizer):
    res = list()
    for tidx, text in enumerate(texts):
        feature = tokenize_text(text, seq_len, tokenizer)
        res.append(feature)
        if tidx < 3:
            tokens, input_ids = feature.tokens, feature.input_ids
            print("*** Example %d ***" % tidx)
            print('\t', len(tokens), '\t', text)
            print('\t', list(zip(tokens, input_ids)))
            print()
    return res


def tokenize_text(text, seq_len, tokenizer):
    text = tokenization.convert_to_unicode(text).strip()
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens[:seq_len - 2] + ["[SEP]"]  # Account for [CLS] and [SEP] with "- 2"
    # if not len(tokens) >= 5:
    #     print('  short text', text)
    # The convention in BERT for single sequences is:
    #    tokens:   [CLS] the dog is hairy . [SEP]
    #    type_ids: 0     0   0   0  0     0 0
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return Tokens(text=text, tokens=tokens, input_ids=input_ids)


class Tokens(object):
    def __init__(self, text, tokens, input_ids):
        self.text = text
        self.tokens = tokens
        self.input_ids = input_ids


class TextExtractor(object):
    def __init__(self):
        self.tokenizer: tokenization.FullTokenizer = None
        self.model: MyBertModel = None
        self.sess: tf.Session = None

    def load_tokenizer(self, vocab_file, do_lower_case=True):
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    def tokenize_text(self, text, seq_len):
        assert self.tokenizer is not None
        return tokenize_text(text, seq_len, self.tokenizer)

    def build_init_and_load(self, bert_config_file, init_checkpoint, gpu_id, gpu_frac):
        from utils.deep.funcs import get_session
        self.model = build_model(bert_config_file, is_training=True)
        self.sess = get_session(gpu_id, gpu_frac, allow_growth=False, run_init=True)
        load_model_params(init_checkpoint)

    def release_resources(self):
        if self.model is None and self.sess is None:
            return
        self.sess.close()
        tf.reset_default_graph()
        del self.model, self.sess
        self.model = self.sess = None

    @staticmethod
    def get_inputs(wids_list, seq_len: int):
        input_ids = list()
        input_mask = list()
        token_type_ids = list()
        for idx, wids in enumerate(wids_list):
            wlen = len(wids)
            input_ids.append(wids + [0] * (seq_len - wlen))
            input_mask.append([1] * wlen + [0] * (seq_len - wlen))
            token_type_ids.append([0] * seq_len)
        return input_ids, input_mask, token_type_ids

    def get_features(self, wids_list, seq_len: int):
        assert self.model is not None and self.sess is not None
        real_inputs = self.get_inputs(wids_list, seq_len)
        model_inputs = [self.model.input_ids, self.model.input_mask, self.model.token_type_ids]
        fd = dict(zip(model_inputs, real_inputs))
        outputs = self.sess.run(self.model.pooled_output, feed_dict=fd)
        return outputs
