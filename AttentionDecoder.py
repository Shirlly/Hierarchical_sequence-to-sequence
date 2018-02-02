import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.rnn import DropoutWrapper


class AttentionDecoder(object):
    def __init__(self, hidden_dim, vocab_size, cell, max_length, decoder_dropout, beam_size=4):
        self.PAD = 0  # End and Pad sequence
        self.GO = 1  # Start of decoding input process
        self.END = 2

        self.decoder_hidden_nodes = hidden_dim
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.cell = cell
        self.max_length = max_length
        self.decoder_dropout = decoder_dropout

        self.decoder_out = tf.placeholder('int32', [None, None], name='decoder_output')
        self.decoder_len = tf.placeholder('int32', [None, ], name='decoder_output_length')

        self.lead = tf.placeholder('int32', [None, None], name='decoder_lead')
        self.lead_out_len = tf.placeholder('int32', [None, ], name='decoder_lead_length')

        self._init_decoder_train_connectors()

    def _init_decoder_train_connectors(self):
        sequence_size, self.batch_size = tf.unstack(tf.shape(self.decoder_out))

        GO_SLICE = tf.ones([1, self.batch_size], dtype=tf.int32) * self.GO
        PAD_SLICE = tf.ones([1, self.batch_size], dtype=tf.int32) * self.PAD

        self.decoder_train_input = tf.concat([GO_SLICE, self.decoder_out], axis=0)
        self.decoder_train_length = self.decoder_len + 1

        decoder_train_targets = tf.concat([self.decoder_out, PAD_SLICE], axis=0)
        decoder_lead = tf.concat([self.lead, PAD_SLICE], axis=0)

        decoder_train_targets_len, _ = tf.unstack(tf.shape(decoder_train_targets))
        decoder_lead_seq_len, _ = tf.unstack(tf.shape(decoder_lead))

        decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                    decoder_train_targets_len,
                                                    on_value=self.END, off_value=self.PAD,
                                                    dtype=tf.int32)

        decoder_lead_eos_mask = tf.one_hot(self.lead_out_len,
                                           decoder_lead_seq_len,
                                           on_value=self.END, off_value=self.PAD,
                                           dtype=tf.int32)

        decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
        decoder_lead_eos_mask = tf.transpose(decoder_lead_eos_mask, [1, 0])

        decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)
        decoder_lead = tf.add(decoder_lead, decoder_lead_eos_mask)

        self.decoder_train_targets = decoder_train_targets
        self.decoder_lead = decoder_lead

        self.loss_weights = tf.ones([self.batch_size, tf.reduce_max(self.decoder_train_length)],
                                    dtype=tf.float32, name='loss_weights')

    def decode(self, cell, helper, scope, sen_encoder_outputs, sen_encoder_len, sen_encoder_final_state,
               de_len, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_states = tf.transpose(sen_encoder_outputs, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.decoder_hidden_nodes, memory=attention_states,
                memory_sequence_length=sen_encoder_len)

            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, alignment_history=True)  # attention_layer_size=self.decoder_hidden_nodes

            if self.decoder_dropout == True:
                attn_cell = DropoutWrapper(cell=attn_cell, output_keep_prob=0.5)
            """
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, self.vocab_size, reuse=reuse
            )
            """

            initial_state = attn_cell.zero_state(dtype=tf.float32, batch_size=de_len)
            initial_state = initial_state.clone(cell_state=sen_encoder_final_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell, helper=helper,
                initial_state=initial_state
                )
            # output_max_length = tf.reduce_max(self.decoder_train_length)
            outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.max_length+1  #  31 #
            )
            return outputs, final_state, final_sequence_lengths

    def process_decoder(self, embeddings, sen_encoder_outputs, sen_encoder_len, sen_encoder_final_state):
        with variable_scope.variable_scope("Decoder") as scope:
            if self.cell == "LSTM":
                cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_nodes)
            else:
                cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_nodes)

            decoder_train_input_embed = tf.nn.embedding_lookup(embeddings, self.decoder_train_input)

            train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_train_input_embed, self.decoder_train_length, time_major=True)

            de_len, _ = tf.unstack(tf.shape(sen_encoder_final_state))
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                   start_tokens=tf.tile([self.GO], [de_len]),
                                                                   end_token=self.PAD)
            sen_en_len = tf.reshape(sen_encoder_len, [-1])

            self.train_out, train_final_state, _ = self.decode(cell, train_helper, 'decode', sen_encoder_outputs, sen_en_len,
                                                          sen_encoder_final_state, de_len)

            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            train_logits = output_fn(self.train_out.rnn_output)
            self.decoder_prediction_train = tf.argmax(train_logits, axis=-1, name='decoder_prediction_train')

            train_attn_weight = train_final_state.alignment_history
            train_attn_weight = nest.map_structure(lambda ta: ta.stack(), train_attn_weight)
            train_attn_wsum = tf.reduce_sum(train_attn_weight, 0)

            """
            infer_attn_weight = infer_final_state.alignment_history
            infer_attn_weight = nest.map_structure(lambda ta: ta.stack(), infer_attn_weight)
            infer_attn_wsum = tf.reduce_sum(infer_attn_weight, 0)
            """

            scope.reuse_variables()
            infer_out, infer_final_state, _ = self.decode(cell, pred_helper, 'decode', sen_encoder_outputs, sen_en_len,
                                                          sen_encoder_final_state, de_len, reuse=True)

            infer_logits = output_fn(infer_out.rnn_output)
            self.decoder_prediction_infer = tf.argmax(infer_logits, axis=-1, name='decoder_prediction_infer')
            return self.decoder_out, self.decoder_len, self.decoder_train_targets, train_logits, infer_logits, \
                   self.loss_weights, self.decoder_prediction_train, self.decoder_prediction_infer, \
                   train_attn_wsum, self.lead, self.lead_out_len, self.decoder_lead, train_final_state #, infer_attn_wsum

    def process_valid(self):
        self.decoder_valid = tf.placeholder('int32', [None, None], name='decoder_valid')
        self.decoder_valid_len = tf.placeholder('int32', [None, ], name='decoder_valid_length')
        PAD_SLICE = tf.ones([1, self.batch_size], dtype=tf.int32) * self.PAD

        decoder_valid_targets = tf.concat([self.decoder_valid, PAD_SLICE], axis=0)
        decoder_valid_targets_len, _ = tf.unstack(tf.shape(decoder_valid_targets))

        decoder_valid_targets_eos_mask = tf.one_hot(self.decoder_valid_len,
                                                    decoder_valid_targets_len,
                                                    on_value=self.PAD, off_value=self.PAD,
                                                    dtype=tf.int32)
        decoder_valid_targets_eos_mask = tf.transpose(decoder_valid_targets_eos_mask, [1, 0])
        decoder_valid_targets = tf.add(decoder_valid_targets, decoder_valid_targets_eos_mask)
        self.decoder_valid_targets = decoder_valid_targets

        self.valid_loss_weights = tf.ones([self.batch_size, tf.reduce_max(self.decoder_valid_len + 1)],
                                          dtype=tf.float32, name='valid_loss_weights')
        return self.decoder_valid, self.decoder_valid_len, self.decoder_valid_targets, self.valid_loss_weights

