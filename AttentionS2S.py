import time
import util
import tensorflow as tf
import numpy as np
from Encoder import Encoder
from AttentionDecoder import AttentionDecoder
from tensorflow.python.ops import variable_scope


class AttentionS2S(object):
    def __init__(self, args, vocab_size, pre_embed, max_length):
        self.PAD = 0
        self.EOS = 1

        with tf.Graph().as_default() as self.graph:
            self.sess = tf.Session(config=util.get_config())
            self.vocab_size = vocab_size
            self.cell = args.cell
            self.mode = args.mode
            self.batch_size = args.batch_size
            self.word_hidden_dim = args.word_hidden_dimension
            self.max_length = max_length
            self.opt = args.opt
            self.decode_mode = args.decode_mode

            # Encoder parameters
            self.embed_dim = args.embed_dim
            self.pre_embed = pre_embed
            self.word_encoder_hidden_nodes = args.word_hidden_dimension
            self.sen_encoder_hidden_nodes = args.sen_hidden_dimension
            self._enc_padding_mask = tf.placeholder(tf.float32, [args.batch_size, None], name='enc_padding_mask')
            self.beam_size = args.beam_size

            # Decoder parameters
            self.decoder_hidden_nodes = self.sen_encoder_hidden_nodes * 2
            self.attention_level = args.attention_level

            # Training parameters
            self.max_grad_norm = args.max_grad_norm
            self.learning_rate = args.learning_rate
            self.adagrad_init_acc = args.adagrad_init_acc

    def simple_cost(self, seq_loss=False):
        de_tar = tf.transpose(self.decoder_train_targets, [1, 0])
        label_in = tf.one_hot(de_tar, depth=self.vocab_size, dtype='float32')
        if seq_loss:
            self._loss = tf.contrib.seq2seq.sequence_loss(
                self.train_logits.rnn_output, self.decoder_train_targets, weights=self.loss_weights)
        else:
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=label_in,
                logits=self.train_logits # .rnn_output
            )
            self._loss = tf.reduce_mean(stepwise_cross_entropy)
        tf.summary.scalar('loss', self._loss)

    def valid_cost(self):
        valid_de_tar = tf.transpose(self.decoder_train_targets, [1, 0])
        valid_label_in = tf.one_hot(valid_de_tar, depth=self.vocab_size, dtype='float32')
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=valid_label_in,
            logits=self.infer_logits # .rnn_output
        )
        self._valid_loss = tf.reduce_mean(stepwise_cross_entropy)
        # tf.summary.scalar('valid_loss', self._valid_loss)

    def train_op(self):
        if self.opt == 'Adagrad':
            tvars = tf.trainable_variables()
            gradients = tf.gradients(self._loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

            grads, global_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)

            tf.summary.scalar('global_norm', global_norm)
            optimizer = tf.train.AdagradOptimizer(self.learning_rate, initial_accumulator_value=self.adagrad_init_acc)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        elif self.opt == 'Adam':
            self._train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self._loss)
        elif self.opt == 'RMS':
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self._loss)


    def build_graph(self):
        # with variable_scope.variable_scope("BuildingGraph") as scope:
        tf.logging.info('Building graph...')
        t0 = time.time()

        encoder = Encoder(self.vocab_size, self.embed_dim, self.pre_embed, self.word_encoder_hidden_nodes,
                          self.sen_encoder_hidden_nodes, self.cell)

        self.word_encoder_input, self.word_encoder_len, self.word_encoder_output, self.word_encoder_final_state, \
        embeddings = encoder.word_encoder()

        self.sen_encoder_len, self.sen_encoder_output, self.sen_encoder_final_state = \
            encoder.sentence_encoder()

        decoder = AttentionDecoder(self.decoder_hidden_nodes, self.vocab_size, self.cell, self.max_length,
                                   self.beam_size)
        if self.decode_mode == "sen_out":
            self.decoder_out, self.decoder_len, self.decoder_train_targets, self.train_logits, self.infer_logits, \
                self.loss_weights, self.decoder_prediction_train, self.decoder_prediction_infer, \
                self.train_attn_wsum, _, _, _, _ = decoder.process_decoder(
                    embeddings, self.sen_encoder_output, self.sen_encoder_len, self.sen_encoder_final_state)
        elif self.decode_mode == "word_out":
            row_len, row_num, _ = tf.unstack(tf.shape(self.word_encoder_output))
            zero_index = tf.where(tf.equal(self.word_encoder_len, 0))
            word_en_len_mask = tf.one_hot(zero_index, row_num, on_value=1, off_value=0, dtype=tf.int32)
            word_en_len_mask = tf.reduce_sum(word_en_len_mask, axis=0)
            self.word_en_de_len = tf.add(self.word_encoder_len, word_en_len_mask)

            self.word_en_output = tf.transpose(self.word_encoder_output, [1, 0, 2])
            clu_num, dim = tf.unstack(tf.shape(self.sen_encoder_final_state))
            self.word_en_output = tf.reshape(self.word_en_output, [clu_num, -1, dim])
            self.word_en_output = tf.transpose(self.word_en_output, [1, 0, 2])

            self.decoder_out, self.decoder_len, self.decoder_train_targets, self.train_logits, self.infer_logits, \
            self.loss_weights, self.decoder_prediction_train, self.decoder_prediction_infer, \
            self.train_attn_wsum = decoder.process_decoder(
                embeddings, self.word_en_output, self.word_en_de_len, self.sen_encoder_final_state)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.mode == 'train':
            self.simple_cost()
            self.train_op()
            # self.valid_cost()

        self._summary = tf.summary.merge_all()
        t1 = time.time()
        tf.logging.info('Time to build graph is: %i seconds', t1 - t0)

    def train_feed_dict_fn(self, data, decoder_target, decoder_len):
        inputs_time_major = data['inputs_time_major']
        sen_len = data['sen_len']
        sen_num = data['sen_num']
        sen_num = np.reshape(np.asarray(sen_num), (-1, 1))
        feed_dict = {self.word_encoder_input: inputs_time_major, self.word_encoder_len: sen_len,
                     self.sen_encoder_len: sen_num, self.decoder_out: decoder_target, self.decoder_len: decoder_len}
        """
        feed_dict = {self.word_encoder_input: data[0], self.word_encoder_len: data[1],
                     self.decoder_out: decoder_target, self.decoder_len: decoder_len}
        """
        return feed_dict

    def to_text(self, idx, vocab_inv):
        text = ""
        for word in idx:
            if word in vocab_inv:
                text = text + " " + vocab_inv[word]
            else:
                text = text + ' UNK'
        return text.strip()

    def run_train_step(self, data, decoder_target, decoder_len):
        feed_dict = self.train_feed_dict_fn(data, decoder_target, decoder_len)
        try:
            return self.sess.run([self._loss, self._train_op, self._summary, self.global_step], feed_dict)
        except:
            print 'Training Wrong'
            import pdb; pdb.set_trace()

    def run_train_result(self, data, decoder_target, decoder_len, vocab_inv):
        feed_dict = self.train_feed_dict_fn(data, decoder_target, decoder_len)
        predict = self.sess.run(self.decoder_prediction_train, feed_dict)
        max_out_num = 10
        for i, (inp, pred) in enumerate(zip(feed_dict[self.decoder_out].T, predict)):
            if i < max_out_num:
                print('sample {}'.format(i + 1))
                inp_text = self.to_text(inp, vocab_inv)
                print('input   > {}'.format(inp_text))
                pred_text = self.to_text(pred, vocab_inv)
                print('predict > {}'.format(pred_text))

    def run_valid_step(self, data, decoder_target, decoder_len):
        feed_dict = self.train_feed_dict_fn(data, decoder_target, decoder_len)
        try:
            valid_pred, train_pred, valid_loss = self.sess.run([self.decoder_prediction_infer, self.decoder_prediction_train,
                                                                self._valid_loss], feed_dict)
            return valid_loss
        except:
            print 'Validation Wrong'
            import pdb; pdb.set_trace()

    def run_valid_result(self, data, decoder_target, decoder_len, vocab_inv):
        feed_dict = self.train_feed_dict_fn(data, decoder_target, decoder_len)
        valid_pred, valid_loss = self.sess.run([self.decoder_prediction_infer, self._valid_loss], feed_dict)
        print '=== Validation loss is :\t', valid_loss
        max_out_num = 2
        for i, (inp, pred) in enumerate(zip(feed_dict[self.decoder_out].T, valid_pred)):
            if i < max_out_num:
                print('sample {}'.format(i + 1))
                inp_text = self.to_text(inp, vocab_inv)
                print('input   > {}'.format(inp_text))
                pred_text = self.to_text(pred, vocab_inv)
                print('predict > {}'.format(pred_text))
            else:
                break

    def run_inference(self, data):
        inputs_time_major = data['inputs_time_major']
        sen_len = data['sen_len']
        sen_num = data['sen_num']
        sen_num = np.reshape(np.asarray(sen_num), (-1, 1))
        feed_dict = {self.word_encoder_input: inputs_time_major, self.word_encoder_len: sen_len,
                     self.sen_encoder_len: sen_num}
        valid_pred = self.sess.run(self.decoder_prediction_infer, feed_dict)
        return valid_pred

    def run_attn_weight(self, data, decoder_target, decoder_len):
        feed_dict = self.train_feed_dict_fn(data, decoder_target, decoder_len)
        attn_weight = self.sess.run(self.train_attn_wsum, feed_dict)
        if self.decode_mode == 'word_out':
            max_sen_num = max(data['sen_num'])
            clu_num = len(data)
            attn_weight = np.reshape(attn_weight, (clu_num, max_sen_num, -1))
            attn_weight = np.sum(attn_weight, axis=2)
        return attn_weight
