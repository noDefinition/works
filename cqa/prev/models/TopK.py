import tensorflow as tf

from prev.models import BasicPair


class TopKSum(BasicPair):
    tc = dict(BasicPair.tc)
    tc['top_k'] = [5]

    def text_rep(self, t):
        t = self.top_k(t, self.args.top_k)
        t = tf.reduce_sum(t, -2)
        return t

    def get_question_rep(self):
        q, q_mask = super().get_que_rep()
        q = self.text_rep(q)
        return q, q_mask

    def rep(self, ans, user):
        q, q_mask = self.question_rep
        ans, a_mask = self.text_embs(ans)  # (bs, al, k)
        ans = self.text_rep(ans)
        user = self.user_emb(user)
        o = self.pair_mul_add([q, ans, user])
        # tf.summary.histogram("rep/q", q)
        # tf.summary.histogram("rep/a", ans)
        # tf.summary.histogram("rep/u", user)
        return o


class TopKMean(TopKSum):
    tc = dict(TopKSum.tc)

    def text_rep(self, t):
        t = self.top_k(t, self.args.top_k)
        t = tf.reduce_mean(t, -2)
        return t
