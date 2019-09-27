import torch.nn as nn

from .transformer import TransformerBlock
from .bert_embedding import BERTEmbedding


class BERT(nn.Module):
    """ BERT model : Bidirectional Encoder Representations from Transformers. """

    def __init__(self, vocab_size, d_hidden=768, n_layers=12, n_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param d_hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param n_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        # paper noted they used (4 * hidden_size) for ff_network_hidden_size
        self.feed_forward_hidden = d_hidden * 4
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_hidden)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_hidden, n_heads, d_hidden * 4, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len])
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)  # (bs, 1, tn, tn)
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x
