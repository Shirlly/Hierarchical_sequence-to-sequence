import os
import util
import time
import math
import argparse
import helpers
import cPickle
import numpy as np
import tensorflow as tf
import tensorgraph as tg
from random import shuffle
from data_add import loop_dir
from helpers import word_encoder_batch, batch
from AttentionS2S import AttentionS2S
from plain_s2s import Seq2Seq
from prepare_cluster_data1 import assignTweet
from preEmbedding import w2v1
from extractSummary import rouge, cosineSim, cosineSimTop
from cluster_process import cluster_demo, grd_news

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", default=64)
parser.add_argument("-d", "--embed_dim", default=300)
parser.add_argument("-l", "--learning_rate", default=0.001)
parser.add_argument("-e", "--max_epoch", default=1000)
parser.add_argument("-cr", "--cluster_rep_restore", default=True)
parser.add_argument("-crd", "--cluster_rep_dir", default='./pre_log_Adam_con')
parser.add_argument("-mr", "--model_restore", default=False)
parser.add_argument("-mrd", "--model_dir", default='./pre_log_Adam_con')
parser.add_argument("-m", "--cluster_method", default='kmeans')
parser.add_argument("-wh", "--word_hidden_dimension", default=128)
parser.add_argument("-sh", "--sen_hidden_dimension", default=128)
parser.add_argument("-c", "--cell", default='GRU')
parser.add_argument("-a", "--attention_level", default='sen')
parser.add_argument("-f", "--log_root", default='./log')
parser.add_argument("--mode", default='train')
parser.add_argument("--decode_mode", default='sen_out', help='sen_out: attention on sentence_encoder_output; '
                                                             'word_out: attention on word_encoder_output')
parser.add_argument("--beam_size", default=4)
parser.add_argument("--max_grad_norm", default=2.0, help='for gradient clipping')
parser.add_argument("--adagrad_init_acc", default=0.1, help='initial accumulator value for Adagrad')
parser.add_argument("--opt", default='Adagrad', help='Optimization method, Adam or Adagrad')
args = parser.parse_args()

PAD = 0

def conv3to2(news_con):
    news2 = []
    news_num = []
    for i in range(len(news_con)):
        nlen = len(news_con[i])
        news_num.append(nlen)
        for j in range(nlen):
            news2.append(news_con[i][j])
    return news2, news_num


def conv2to3(tweets, twee_num, news_con):
    twee3 = []
    for i in range(len(news_con)):
        snum = sum(twee_num[:i])
        enum = snum + twee_num[i]
        twee3.append(tweets[snum:enum])
    return twee3


def next_feed_inference(news, encoder_input, en_in_len):
    _encoder, _encoder_len, _ = helpers.batch(news)
    return {encoder_input: _encoder, en_in_len: _encoder_len}


def cluster_data(vocab_size, pre_embed, news_con, tweets, twe_num, summary, vocab_inv):
    news2, news_num = conv3to2(news_con)
    tweet3 = conv2to3(tweets, twe_num, news_con)
    s2s = Seq2Seq(vocab_size, args.embed_dim, pre_embed, 128, 256, args.cell, args.beam_size, 30)

    with s2s.graph.as_default():
        encoder_input, en_in_len, encoder_output, encoder_final_state = s2s.encoder()
        saver = tf.train.Saver()
        if args.cluster_rep_restore == True:
            print 'Cluster restore True'
            try:
                saver.restore(s2s.sess, args.cluster_rep_dir + '/best_epoch_checkpoint-31')
            except:
                import pdb; pdb.set_trace()
        else:
            print 'Cluster restore False'
            tf.set_random_seed(1)
            init = tf.global_variables_initializer()
            s2s.sess.run(init)

        feed_dict = next_feed_inference(news2, encoder_input, en_in_len)
        news_state = s2s.sess.run(encoder_final_state, feed_dict)
        sen_rep = news_state  # news_state.h  #
        news_rep = sen_rep.tolist()
        news3 = conv2to3(news_rep, news_num, news_con)

        feed_dict = next_feed_inference(tweets, encoder_input, en_in_len)
        twee_state = s2s.sess.run(encoder_final_state, feed_dict)
        sen_rep = twee_state  # twee_state.h  #
        twee_rep = sen_rep.tolist()
        twee3 = conv2to3(twee_rep, twe_num, news_con)

    if len(news_num) == len(twe_num) == len(summary):
        cluster = cluster_demo(news3, news_num, summary)
        print 'Clustering Done'

        encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info, clu_sen_len \
            = assignTweet(cluster, news3, twee3, tweet3, news_con, news_num, summary, vocab_inv)

        return encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info
    else:
        raise Exception('Number not matching !!')


def run_training(s2s, train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar, vocab_inv, max_twee_len,
                 decoder_infer_tar, decoder_infer_pin, summary, news_con):
    with s2s.graph.as_default():
        train_dir = os.path.join(args.log_root + '_' + str(args.model_restore) + '_' + args.opt + 'hie', "train")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        s2s.build_graph()
        saver = tf.train.Saver()
        if args.model_restore:
            saver.restore(s2s.sess, args.model_dir + '/best_epoch_checkpoint-31')
        else:
            tf.set_random_seed(1)
            init = tf.global_variables_initializer()
            s2s.sess.run(init)
        summary_writer = tf.summary.FileWriter(train_dir + '/cost', s2s.sess.graph)
        data_train = tg.SequentialIterator(train_encoder_in, train_decoder_tar, batchsize=int(args.batch_size))
        data_valid = tg.SequentialIterator(valid_encoder_in, valid_decoder_tar, batchsize=int(args.batch_size))

        steps = 0
        min_valid_loss = float("inf")
        optimal_step = 0
        opt = False
        min_epoch_loss = float("inf")
        min_epoch = 0
        for epoch in range(int(args.max_epoch)):
            print('epoch: ', epoch)
            print('..training')
            loss_epoch = []
            for news_batch, tweet_batch in data_train:
                data = word_encoder_batch(news_batch)
                decoder_target, decoder_len, _ = batch(tweet_batch)  # , max_twee_len
                loss, train_op, summ, global_step = s2s.run_train_step(data, decoder_target, decoder_len)
                loss_epoch.append(loss)
                summary_writer.add_summary(summ, global_step)
                steps += 1
                if steps % 100 == 0 and steps != 0:
                    summary_writer.flush()
                    s2s.run_train_result(data, decoder_target, decoder_len, vocab_inv)
                    valid_loss = []
                    print 'Step {} for validation '.format(steps)

                    for news_valid, tweet_valid in data_valid:
                        data = word_encoder_batch(news_valid)
                        decoder_target, decoder_len, _ = batch(tweet_valid, max_twee_len)  #
                        valid_loss.append(s2s.run_valid_step(data, decoder_target, decoder_len))
                    s2s.run_valid_result(data, decoder_target, decoder_len, vocab_inv)
                    if sum(valid_loss) < min_valid_loss:
                        min_valid_loss = sum(valid_loss)
                        optimal_step = steps
                        saver.save(s2s.sess, train_dir + '/best_step_checkpoint')
                        print 'Saving model'
                    if (steps - optimal_step) % 100 > 10:
                        opt = True
                        break

                if opt:
                    break
            if opt:
                break

            print('epoch loss: {}'.format(sum(loss_epoch)))
            if sum(loss_epoch) < min_epoch_loss:
                min_epoch_loss = sum(loss_epoch)
                min_epoch = epoch
                saver.save(s2s.sess, train_dir + '/best_epoch_checkpoint')
                print 'Saving model'
                # run_inference(s2s, decoder_infer_tar, decoder_infer_pin, summary, news_con, vocab_inv)
        summary_writer.close()

        print("Best running step is ", optimal_step)
        print("Minimum validation loss is ", min_valid_loss)
        print("*** Running end after {} epochs and {} iterations!!! ***".format(epoch, steps))

        print("*** The best model tested by validation data achieved at {} step ! ***".format(min_epoch))   # optimal_step
    return train_dir


def calculate_news_weight(s2s, encoder_in, decoder_tar, data_info, news_grd_sim_rank, max_twee_len):
    with s2s.graph.as_default():
        s2s.build_graph()
        saver = tf.train.Saver()
        saver.restore(s2s.sess, args.log_root + '/best_checkpoint-0')

        data_tlt = tg.SequentialIterator(encoder_in, decoder_tar, batchsize=128)
        attn_weight_tlt = []
        for news_batch, tweet_batch in data_tlt:
            data = word_encoder_batch(news_batch)
            decoder_target, decoder_len, _ = batch(tweet_batch)  # , max_twee_len
            attn_weight = s2s.run_attn_weight(data, decoder_target, decoder_len)
            attn_weight_tlt.append(attn_weight.tolist())

        weight_dict = {}
        news_idx_clu = {}
        try:
            assert len(data_info) == len(attn_weight_tlt)
        except:
            import pdb; pdb.set_trace()
        for i in range(len(attn_weight_tlt)):
            key = data_info[i].keys()[0]
            if key in weight_dict:
                value = weight_dict[key]
                value.append(attn_weight_tlt[i])
                weight_dict[key] = value
            else:
                value = []
                weight_dict[key] = value.append(attn_weight_tlt[i])
                news_idx_clu[key] = data_info[i][key]

        top3 = 0
        base = 0
        tlt = 0
        avg_weight_dict = {}
        twee_top_news = {}
        for key in weight_dict.keys():
            value = np.asarray(weight_dict[key])
            clu_twee_num, clu_news_num = value.shape
            if clu_twee_num > 0:
                clu_avg_weight = np.divide(np.sum(value, axis=0), clu_twee_num*1.0)
                avg_weight_dict[key] = clu_avg_weight
                news_id = [m[0] for m in sorted(enumerate(clu_avg_weight), key=lambda x: x[1], reverse=True)]
                top_id = news_id[:3] if len(news_id) > 3 else news_id
                twee_top_news[key] = top_id
                doc_id, _ = key.split(",")
                for ele in top_id:
                    rk = min(news_grd_sim_rank[doc_id][ele])
                    if rk < 3:
                        top3 += 1
                tlt += len(top_id)
                base += 3
            else:
                twee_top_news[key] = []

        print "== Precision for tweets vote top 3 news is ", (top3 * 1.0)/tlt
        print "== Recall for tweets vote top 3 news is ", (top3 * 1.0)/base
    return twee_top_news


def run_inference(s2s, decoder_infer_tar, decoder_infer_pin, summary, news_con, vocab_inv):
    data_infer = tg.SequentialIterator(decoder_infer_tar, decoder_infer_pin, batchsize=500)
    for news_batch, newsc in data_infer:
        data = word_encoder_batch(news_batch)
        predict = s2s.run_inference(data)
        # evaluate_valid_rouge(summary, predict, news_con, vocab_inv, max_prf)
        write_prediction(predict, summary, s2s, vocab_inv)


def write_prediction(predict, summary, s2s, vocab_inv):
    pred_dir = args.log_root + '/best_valid_predict/'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    predict = predict.tolist()
    tlt_num = 0
    for i in range(len(summary)):
        text_dir = os.path.join(pred_dir, str(i) + '.txt')
        text_s = 'Summary: \n'
        text_p = 'Prediction: \n'
        for j in range(len(summary[i])):
            text_s += s2s.to_text(summary[i][j], vocab_inv) + "\n"
            text_p += s2s.to_text(predict[tlt_num + j], vocab_inv) + "\n"
        tlt_num += len(summary[i])
        text = text_s + "\n" + text_p
        cPickle.dump(text, open(text_dir, 'wb'))


def evaluate_valid_rouge(summary, predict, news_con, vocab_inv, max_prf):
    pnum = 0
    pred_eva = []
    pred_sig = []
    news_tfidf, pred_tfidf = [], []
    predt = predict.T
    for k in range(len(summary)):
        pred = []
        preds = []
        temp = []
        sumc = []
        for m in range(len(summary[k])):
            pred += predt[pnum + m].tolist()
            preds.append(predt[pnum + m].tolist())
            sumc += summary[k][m]
        pred_eva.append([pred])
        pred_sig.append(preds)
        temp.append(sumc)
        pnum += len(summary[k])

    extr_sum, reference = cosineSim(news_con, pred_eva, news_tfidf, pred_tfidf, vocab_inv, summary)
    print 'Concatenate summary:'
    result_cat = rouge(extr_sum, reference)
    extr_sum1, reference, extr_sum_top3 = cosineSimTop(news_con, pred_sig, news_tfidf, pred_tfidf,
                                                       vocab_inv, summary)
    print 'Extract 3 rouge:'
    result_t3 = rouge(extr_sum_top3, reference)
    print 'Extract 1 rouge:'
    result_t1 = rouge(extr_sum1, reference)
    if result_cat['ROUGE-1-F'] > max_prf['max_rf_cat']:
        max_prf['max_rf_cat'] = result_cat['ROUGE-1-F']
        max_prf['max_rp_cat'] = result_cat['ROUGE-1-P']
        max_prf['max_rr_cat'] = result_cat['ROUGE-1-R']
        max_prf['max_pred_cat'] = predt.tolist()
    if result_t1['ROUGE-1-F'] > max_prf['max_rf_t1']:
        max_prf['max_rf_t1'] = result_t1['ROUGE-1-F']
        max_prf['max_rp_t1'] = result_t1['ROUGE-1-P']
        max_prf['max_rr_t1'] = result_t1['ROUGE-1-R']
        max_prf['max_pred_t1'] = predt.tolist()
    if result_t3['ROUGE-1-F'] > max_prf['max_rf_t3']:
        max_prf['max_rf_t3'] = result_t3['ROUGE-1-F']
        max_prf['max_rp_t3'] = result_t3['ROUGE-1-P']
        max_prf['max_rr_t3'] = result_t3['ROUGE-1-R']
        max_prf['max_pred_t3'] = predt.tolist()
    return max_prf


def evaluation_data(encoder_in, decoder_tar):

    data = list(zip(encoder_in, decoder_tar))
    shuffle(data)
    encoder_in, decoder_tar = zip(*data)

    train_num = int(math.ceil(len(encoder_in) * 0.8))
    train_encoder_in = encoder_in[:train_num]
    train_decoder_tar = decoder_tar[:train_num]
    eval_encoder_in = encoder_in[train_num:]
    eval_decoder_tar = decoder_tar[train_num:]
    return train_encoder_in, train_decoder_tar, eval_encoder_in, eval_decoder_tar


def main():
    summary, news, news_twee, tweets, vocab, vocab_inv, max_twee_len, news_con, title, first, \
        twe_num, news_num1, sum_org, news_org, tweet_org = loop_dir()
    vocab_size = len(vocab)
    print 'Vocabulary size is ', vocab_size
    max_twee_len = max_twee_len + 3
    top_news_grd, num3, rest, news_grd_sim_value, news_grd_sim_rank = grd_news(sum_org, news_org)
    pre_embed, _ = w2v1(vocab)

    encoder_in, decoder_tar, decoder_infer_tar, decoder_infer_pin, data_info = \
        cluster_data(vocab_size, pre_embed, news_con, tweets, twe_num, summary, vocab_inv)

    train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar = evaluation_data(encoder_in, decoder_tar)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting seq2seq_attention in %s mode...', args.mode)

    tf.set_random_seed(1)
    s2s = AttentionS2S(args, vocab_size, pre_embed, max_twee_len)

    if args.mode == 'train':
        run_training(s2s, train_encoder_in, train_decoder_tar, valid_encoder_in, valid_decoder_tar, vocab_inv, max_twee_len,
                     decoder_infer_tar, decoder_infer_pin, summary, news_con)
        # run_inference(s2s, decoder_infer_tar, decoder_infer_pin, summary, news_con, vocab_inv)
        # calculate_news_weight(s2s, encoder_in, decoder_tar, data_info, news_grd_sim_rank, max_twee_len)
    elif args.mode == 'inference':
        run_inference(s2s, decoder_infer_tar, decoder_infer_pin, summary, news_con, vocab_inv)
        calculate_news_weight(s2s, encoder_in, decoder_tar, data_info, news_grd_sim_rank, max_twee_len)

if __name__ == '__main__':
    main()
