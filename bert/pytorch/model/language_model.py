import torch.nn as nn
from torch.optim import Adam

from bert.pytorch.model.bert import BERT
from bert.pytorch.trainer.optim_schedule import ScheduledOptim


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size: int, learning_rate: float):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """
        super().__init__()
        self.bert = bert
        self.learning_rate = learning_rate
        self.next_pred = NextSentencePrediction(self.bert.d_hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.d_hidden, vocab_size)
        self.next_pred_nll = nn.NLLLoss()
        self.mask_lm_nll = nn.NLLLoss(ignore_index=0)
        self.optim = Adam(
            params=self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        self.schedule = ScheduledOptim(
            optimizer=self.optim,
            d_model=self.bert.d_hidden,
            n_warmup_steps=10000,
        )

    def forward(self, x, segment):
        x = self.bert(x, segment)  # (bs, tn, dw)
        return self.next_pred(x[:, 0]), self.mask_lm(x)

    def train_step(self, wints, labels, segment, next_real):
        # 1. forward the next_sentence_prediction and masked_lm model
        next_pred, mask_lm_pred = self(wints, segment)  # (bs, 2), (bs, tn, nw)

        # 2-1. NLL(negative log likelihood) loss of is_next classification result
        next_loss = self.next_pred_nll(next_pred, next_real)
        # 2-2. NLLLoss of predicting masked token word
        mask_loss = self.mask_lm_nll(mask_lm_pred.transpose(1, 2), labels)
        # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
        loss = next_loss + mask_loss

        # 3. backward and optimization only in train
        self.schedule.zero_grad()
        loss.backward()
        self.schedule.step_and_update_lr()
        # self.optim.zero_grad()
        # loss.backward()
        # self.optim.step()

        # next sentence prediction accuracy
        correct_cnt = next_pred.argmax(dim=1).eq(next_real).sum().item()
        loss_val = loss.item()
        return correct_cnt, loss_val


class NextSentencePrediction(nn.Module):
    """ 2-class classification model : is_next, is_not_next """

    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
