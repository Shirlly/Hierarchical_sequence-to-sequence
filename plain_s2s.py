import util
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn import GRUCell, LSTMCell, LSTMStateTuple
from tensorflow.python.util import nest


class Seq2Seq(object):
    def __init__(self, vocab_size, embed_dim, pre_embed, encoder_hidden_dim, hidden_dim,
                 cell, beam_size, max_len):
        with tf.Graph().as_default() as self.graph:
            self.sess = tf.Session()
            self.PAD = 0  # Pad sequence
            self.GO = 1  # Start of decoding input process
            self.END = 2  # End of sequence

            self.cell = cell
            self.max_len = max_len

            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.embed = tf.Variable(tf.random_uniform([self.vocab_size + 4, self.embed_dim], -1.0, 1.0),
                                     'float32', name='embedding')
            self.pre_embed = pre_embed
            self._pre_embed_replace()

            self.encoder_hidden_nodes = encoder_hidden_dim
            self.encoder_input = tf.placeholder('int32', [None, None],
                                                name='word_encoder_input')
            self.encoder_len = tf.placeholder('int32', [None, ], name='word_encoder_input_length')

            self.decoder_hidden_nodes = hidden_dim
            self.beam_size = beam_size

            self.decoder_out = tf.placeholder('int32', [None, None], name='decoder_output')
            self.decoder_len = tf.placeholder('int32', [None, ], name='decoder_output_length')

            self._init_decoder_train_connectors()

    def _pre_embed_replace(self):
        for ele in self.pre_embed:
            self.embed[ele].assign(self.pre_embed[ele])

    def _init_decoder_train_connectors(self):
        sequence_size, self.batch_size = tf.unstack(tf.shape(self.decoder_out))

        GO_SLICE = tf.ones([1, self.batch_size], dtype=tf.int32) * self.GO
        PAD_SLICE = tf.ones([1, self.batch_size], dtype=tf.int32) * self.PAD

        self.decoder_train_input = tf.concat([GO_SLICE, self.decoder_out], axis=0)
        self.decoder_train_length = self.decoder_len + 1

        decoder_train_targets = tf.concat([self.decoder_out, PAD_SLICE], axis=0)

        decoder_train_targets_len, _ = tf.unstack(tf.shape(decoder_train_targets))

        decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                    decoder_train_targets_len,
                                                    on_value=self.END, off_value=self.PAD,
                                                    dtype=tf.int32)

        decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])

        decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)

        self.decoder_train_targets = decoder_train_targets

        self.loss_weights = tf.ones([self.batch_size, tf.reduce_max(self.decoder_train_length)],
                                    dtype=tf.float32, name='loss_weights')

    def encoder(self):
        with self.graph.as_default():
            with variable_scope.variable_scope("Encoder") as scope:
                if self.cell is 'GRU':
                    encoder_cell = GRUCell(self.encoder_hidden_nodes)
                elif self.cell is 'LSTM':
                    encoder_cell = LSTMCell(self.encoder_hidden_nodes)

                encoder_input_embed = tf.nn.embedding_lookup(self.embed, self.encoder_input)

                ((encoder_fw_output, encoder_bw_output),
                 (encoder_fw_final_state, encoder_bw_final_state)) \
                    = (tf.nn.bidirectional_dynamic_rnn
                       (cell_fw=encoder_cell, cell_bw=encoder_cell,
                        inputs=encoder_input_embed, time_major=True,
                        sequence_length=self.encoder_len, dtype=tf.float32))

                self.encoder_output = tf.concat((encoder_fw_output, encoder_bw_output), 2)

                if self.cell is 'GRU':
                    self.encoder_final_state = tf.concat((encoder_fw_final_state, encoder_bw_final_state), 1)
                elif self.cell is 'LSTM':
                    encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                    encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
                    self.encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

            return self.encoder_input, self.encoder_len, self.encoder_output, self.encoder_final_state

    def decode(self, cell, helper, scope, sen_encoder_outputs, sen_encoder_len,
               sen_encoder_final_state, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_states = tf.transpose(sen_encoder_outputs, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.decoder_hidden_nodes, memory=attention_states,
                memory_sequence_length=sen_encoder_len)

            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, alignment_history=True)

            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, self.vocab_size, reuse=reuse
            )

            initial_state = out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
            initial_state = initial_state.clone(cell_state=sen_encoder_final_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=initial_state
                )

            outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=self.max_len + 3
            )
            return outputs, final_state, final_sequence_lengths

    def process_decoder(self):
        with self.graph.as_default():
            with variable_scope.variable_scope("Decoder") as scope:
                if self.cell == "LSTM":
                    cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_nodes)
                else:
                    cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_nodes)

                decoder_train_input_embed = tf.nn.embedding_lookup(self.embed, self.decoder_train_input)

                train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_train_input_embed, self.decoder_train_length,
                                                                 time_major=True)

                pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed,
                                                                       start_tokens=tf.tile([self.GO],
                                                                                            [self.batch_size]),
                                                                       end_token=self.PAD)

                train_logits, train_final_state, _ = self.decode(cell, train_helper, 'decode', self.encoder_output,
                                                                 self.encoder_len, self.encoder_final_state)

                def output_fn(outputs):
                    return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

                # train_logits = output_fn(train_out.rnn_output)
                self.decoder_prediction_train = tf.argmax(train_logits.rnn_output, axis=-1, name='decoder_prediction_train')

                train_attn_weight = train_final_state.alignment_history
                train_attn_weight = nest.map_structure(lambda ta: ta.stack(), train_attn_weight)
                train_attn_wsum = tf.reduce_sum(train_attn_weight, 0)

                """
                infer_attn_weight = infer_final_state.alignment_history
                infer_attn_weight = nest.map_structure(lambda ta: ta.stack(), infer_attn_weight)
                infer_attn_wsum = tf.reduce_sum(infer_attn_weight, 0)
                """

                scope.reuse_variables()
                infer_logits, infer_final_state, _ = self.decode(cell, pred_helper, 'decode', self.encoder_output,
                                                                 self.encoder_len, self.encoder_final_state, reuse=True)
                # infer_logits = output_fn(infer_out.rnn_output)
                self.decoder_prediction_infer = tf.argmax(infer_logits.rnn_output, axis=-1, name='decoder_prediction_infer')
                return self.decoder_out, self.decoder_len, self.decoder_train_targets, train_logits.rnn_output, infer_logits, \
                       self.loss_weights, self.decoder_prediction_train, self.decoder_prediction_infer, \
                       train_attn_wsum# , infer_attn_wsum

