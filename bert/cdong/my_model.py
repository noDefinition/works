from bert.modeling import *


class MyBertModel(object):
    def __init__(self, config, is_training):
        """Constructor for BertModel.

        Args:
          config: `BertConfig` instance.
          is_training: bool. true for training model, false for eval model. Controls
            whether dropout will be applied.
            embeddings or tf.embedding_lookup() for the word embeddings.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_ids = self.input_ids = tf.placeholder(tf.int32, shape=(None, None))
        input_mask = self.input_mask = tf.placeholder(tf.int32, shape=(None, None))
        token_type_ids = self.token_type_ids = tf.placeholder(tf.int32, shape=(None, None))

        with tf.variable_scope("bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=False)
                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)
                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.nn.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_embedding_output(self):
        return self.embedding_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_table(self):
        return self.embedding_table

    def get_feed_dict(self, input_ids, input_mask, token_type_ids):
        input_shape = input_ids.shape
        if input_mask is None:
            input_mask = np.ones(shape=input_shape, dtype=np.int32)
        else:
            assert input_mask.shape == input_shape
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=input_shape, dtype=np.int32)
        else:
            assert token_type_ids.shape == input_shape
        return {
            self.input_ids: input_ids,
            self.input_mask: input_mask,
            self.token_type_ids: token_type_ids}
