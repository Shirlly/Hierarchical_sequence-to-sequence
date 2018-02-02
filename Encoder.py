import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn import GRUCell, LSTMCell, LSTMStateTuple, DropoutWrapper

PAD = 0


class Encoder(object):
    def __init__(self, vocab_size, embed_dim, pre_embed, word_hidden_dim, sen_hidden_dim, cell, encoder_dropout=True):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.cell = cell
        self.encoder_dropout = encoder_dropout
        self.word_encoder_hidden_nodes = word_hidden_dim
        self.sen_encoder_hidden_nodes = sen_hidden_dim

        self.pre_embed = pre_embed
        self.embed = tf.Variable(tf.random_uniform([self.vocab_size + 4, self.embed_dim], -1.0, 1.0),
                                 'float32', name='embedding')
        self._pre_embed_replace()

        self.word_encoder_input = tf.placeholder('int32', [None, None],
                                                 name='word_encoder_input')  # [clu_num*max_sen_num, sen_len]
        self.word_encoder_len = tf.placeholder('int32', [None, ], name='word_encoder_input_length') # [sen_len]
        self.sen_encoder_len = tf.placeholder('int32', [None, 1], name='sen_encoder_input_length')

    def _pre_embed_replace(self):
        for ele in self.pre_embed:
            self.embed[ele].assign(self.pre_embed[ele])

    def word_encoder(self):
        with variable_scope.variable_scope("Encoder") as scope:
            if self.cell is 'GRU':
                encoder_cell = GRUCell(self.word_encoder_hidden_nodes)
                if self.encoder_dropout == True:
                    encoder_cell = DropoutWrapper(cell=encoder_cell, output_keep_prob=0.5)
            elif self.cell is 'LSTM':
                encoder_cell = LSTMCell(self.word_encoder_hidden_nodes)
                if self.encoder_dropout == True:
                    encoder_cell = DropoutWrapper(cell=encoder_cell, output_keep_prob=0.5)

            encoder_input_embed = tf.nn.embedding_lookup(self.embed, self.word_encoder_input)

            ((encoder_fw_output, encoder_bw_output),
                (encoder_fw_final_state, encoder_bw_final_state)) \
                = (tf.nn.bidirectional_dynamic_rnn
                   (cell_fw=encoder_cell, cell_bw=encoder_cell,
                    inputs=encoder_input_embed, time_major=True,
                    sequence_length=self.word_encoder_len, dtype=tf.float32))

            self.word_encoder_output = tf.concat((encoder_fw_output, encoder_bw_output), 2)

            if self.cell is 'GRU':
                self.word_encoder_final_state = tf.concat((encoder_fw_final_state, encoder_bw_final_state), 1)
            elif self.cell is 'LSTM':
                encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
                self.word_encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

        return self.word_encoder_input, self.word_encoder_len, self.word_encoder_output, self.word_encoder_final_state,\
               self.embed

    def sentence_encoder(self):
        with variable_scope.variable_scope("SenEncoder") as scope:
            if self.cell is 'GRU':
                encoder_cell = GRUCell(self.sen_encoder_hidden_nodes)
            elif self.cell is 'LSTM':
                encoder_cell = LSTMCell(self.sen_encoder_hidden_nodes)

            max_sen_num = tf.reduce_max(self.sen_encoder_len)

            clu_num, _ = tf.unstack(tf.shape(self.sen_encoder_len))
            sen_encoder_len = tf.reshape(self.sen_encoder_len, [-1])

            sen_encoder_input = tf.reshape(self.word_encoder_final_state,
                                           [clu_num, max_sen_num, self.word_encoder_hidden_nodes*2])
            sen_encoder_input = tf.transpose(sen_encoder_input, [1, 0, 2])
            # max_sen_num *clu_num * hidden_dim

            ((encoder_fw_output, encoder_bw_output),
             (encoder_fw_final_state, encoder_bw_final_state)) \
                = (tf.nn.bidirectional_dynamic_rnn
                   (cell_fw=encoder_cell, cell_bw=encoder_cell,
                    inputs=sen_encoder_input, time_major=True,
                    sequence_length=sen_encoder_len, dtype=tf.float32))

            sen_encoder_output = tf.concat((encoder_fw_output, encoder_bw_output), 2)

            if self.cell is 'GRU':
                sen_encoder_final_state = tf.concat((encoder_fw_final_state, encoder_bw_final_state), 1)
            elif self.cell is 'LSTM':
                encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
                encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
                sen_encoder_final_state = LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)

        return self.sen_encoder_len, sen_encoder_output, sen_encoder_final_state

