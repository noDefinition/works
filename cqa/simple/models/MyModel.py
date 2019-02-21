import tensorflow as tf

from . import BasicPair


class test(BasicPair):
    def rep(self, que, ans, user, features):
        a, a_mask = self.text_embs(ans)
        a = self.mask_mean(a, a_mask)
        u = self.user_emb(user)
        return a + u


class TopK_and_User(BasicPair):
    def rep(self, que, ans, user, features):
        a, a_mask = self.text_embs(ans)
        a = self.mask_top_k_mean(a, a_mask, 5)
        u = self.user_emb(user)
        return tf.concat([a, u], -1)


class TopK_User_Features(BasicPair):
    def rep(self, que, ans, user, features):
        a, a_mask = self.text_embs(ans)
        a = self.mask_top_k_mean(a, a_mask, 5)
        u = self.user_emb(user)
        f = self.features_rep(features, dim_k=20)
        return tf.concat([a, u, f], -1)


# class AUF_add(BasicPair):
#     using_data = {'ans', 'user', 'features'}
#
#     def rep(self, que, ans, user, features):
#         a, a_mask = self.text_emb(ans)
#         a = self.mask_top_k_mean(a, a_mask, 5)
#         u = self.user_emb(user)
#         f = self.features_rep(features, dim_k=20)
#         o = 0.
#         if 'ans' in self.using_data:
#             o += self.top_layers(a, name='AnsTop')
#         if 'user' in self.using_data:
#             o += self.top_layers(u, name='UserTop')
#         if 'features' in self.using_data:
#             o += self.top_layers(f, fc=[], name='FeaturesTop')
#         return o
#
#
# class Ans(AUF_add):
#     using_data = {'ans'}
#
#
# class User(AUF_add):
#     using_data = {'user'}
#
#
# class Features(AUF_add):
#     using_data = {'features'}
#
#
# class AU(AUF_add):
#     using_data = {'ans', 'user'}
#
#
# class AF(AUF_add):
#     using_data = {'ans', 'features'}
#
#
# class UF(AUF_add):
#     using_data = {'user', 'features'}
