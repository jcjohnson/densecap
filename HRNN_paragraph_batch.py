#!/usr/bin/env python
# coding=utf-8

__author__ = "Xinpeng.Chen"

import os
import sys
import time
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import pandas as pd
import random

import h5py
import ipdb

import tensorflow as tf


# ------------------------------------------------------------------------------------------------------
# Initialization class
#  1. Pooling the visual features into a single dense feature
#  2. Then, build sentence LSTM, word LSTM
# ------------------------------------------------------------------------------------------------------
class RegionPooling_HierarchicalRNN():
    def __init__(self, n_words,
                       batch_size,
                       num_boxes,
                       feats_dim,
                       project_dim,
                       sentRNN_lstm_dim,
                       sentRNN_FC_dim,
                       wordRNN_lstm_dim,
                       S_max,
                       N_max,
                       word_embed_dim,
                       bias_init_vector=None):

        self.n_words = n_words
        self.batch_size = batch_size
        self.num_boxes = num_boxes # 50
        self.feats_dim = feats_dim # 4096
        self.project_dim = project_dim # 1024
        self.S_max = S_max # 6
        self.N_max = N_max # 50
        self.word_embed_dim = word_embed_dim # 1024

        self.sentRNN_lstm_dim = sentRNN_lstm_dim # 512 hidden size
        self.sentRNN_FC_dim = sentRNN_FC_dim # 1024 in fully connected layer
        self.wordRNN_lstm_dim = wordRNN_lstm_dim # 512 hidden size

        # word embedding, parameters of embedding
        # embedding shape: n_words x wordRNN_lstm_dim
        with tf.device('/cpu:0'):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, word_embed_dim], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([word_embed_dim]), name='bemb')

        # regionPooling_W shape: 4096 x 1024
        # regionPooling_b shape: 1024
        self.regionPooling_W = tf.Variable(tf.random_uniform([feats_dim, project_dim], -0.1, 0.1), name='regionPooling_W')
        self.regionPooling_b = tf.Variable(tf.zeros([project_dim]), name='regionPooling_b')

        # sentence LSTM
        self.sent_LSTM = tf.nn.rnn_cell.BasicLSTMCell(sentRNN_lstm_dim, state_is_tuple=True)

        # logistic classifier
        self.logistic_Theta_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, 2], -0.1, 0.1), name='logistic_Theta_W')
        self.logistic_Theta_b = tf.Variable(tf.zeros(2), name='logistic_Theta_b')

        # fc1_W: 512 x 1024, fc1_b: 1024
        # fc2_W: 1024 x 1024, fc2_b: 1024
        self.fc1_W = tf.Variable(tf.random_uniform([sentRNN_lstm_dim, sentRNN_FC_dim], -0.1, 0.1), name='fc1_W')
        self.fc1_b = tf.Variable(tf.zeros(sentRNN_FC_dim), name='fc1_b')
        self.fc2_W = tf.Variable(tf.random_uniform([sentRNN_FC_dim, 1024], -0.1, 0.1), name='fc2_W')
        self.fc2_b = tf.Variable(tf.zeros(1024), name='fc2_b')

        # word LSTM
        self.word_LSTM = tf.nn.rnn_cell.BasicLSTMCell(wordRNN_lstm_dim, state_is_tuple=True)
        self.word_LSTM = tf.nn.rnn_cell.MultiRNNCell([self.word_LSTM] * 2, state_is_tuple=True)
        #self.word_LSTM2 = tf.nn.rnn_cell.BasicLSTMCell(wordRNN_lstm_dim, state_is_tuple=True)

        self.embed_word_W = tf.Variable(tf.random_uniform([wordRNN_lstm_dim, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        # receive the feats in the current image
        # it's shape is 10 x 50 x 4096
        # tmp_feats: 500 x 4096
        feats = tf.placeholder(tf.float32, [self.batch_size, self.num_boxes, self.feats_dim])
        tmp_feats = tf.reshape(feats, [-1, self.feats_dim])

        # project_vec_all: 500 x 4096 * 4096 x 1024 --> 500 x 1024
        # project_vec: 10 x 1024
        project_vec_all = tf.matmul(tmp_feats, self.regionPooling_W) + self.regionPooling_b
        project_vec_all = tf.reshape(project_vec_all, [self.batch_size, 50, self.project_dim])
        project_vec = tf.reduce_max(project_vec_all, reduction_indices=1)

        # receive the [continue:0, stop:1] lists
        # example: [0, 0, 0, 0, 1, 1], it means this paragraph has five sentences
        num_distribution = tf.placeholder(tf.int32, [self.batch_size, self.S_max])

        # receive the ground truth words, which has been changed to idx use word2idx function
        captions = tf.placeholder(tf.int32, [self.batch_size, self.S_max, self.N_max+1])
        captions_masks = tf.placeholder(tf.float32, [self.batch_size, self.S_max, self.N_max+1])

        # ---------------------------------------------------------------------------------------------------------------------
        # The method which initialize the state, is refered from below sites:
        # 1. http://stackoverflow.com/questions/38241410/tensorflow-remember-lstm-state-for-next-batch-stateful-lstm/38417699
        # 2. https://www.tensorflow.org/api_docs/python/rnn_cell/classes_storing_split_rnncell_state#LSTMStateTuple
        # 3. https://medium.com/@erikhallstrm/using-the-tensorflow-lstm-api-3-7-5f2b97ca6b73#.u4w9z6h0h
        # ---------------------------------------------------------------------------------------------------------------------
        sent_state = self.sent_LSTM.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        #word_state = self.word_LSTM.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        #word_state1 = self.word_LSTM1.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        #word_state2 = self.word_LSTM2.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        #sent_state = tf.zeros([self.batch_size, self.sent_LSTM1.state_size])
        #word_state1 = tf.zeros([self.batch_size, self.word_LSTM1.state_size])
        #word_state2 = tf.zeros([self.batch_size, self.word_LSTM2.state_size])

        probs = []
        loss = 0.0
        loss_sent = 0.0
        loss_word = 0.0
        lambda_sent = 5.0
        lambda_word = 1.0

        print 'Start build model:'
        #----------------------------------------------------------------------------------------------
        # Hierarchical RNN: sentence RNN and words RNN
        # The word RNN has the max number, N_max = 50, the number in the papar is 50
        #----------------------------------------------------------------------------------------------
        for i in range(0, self.S_max):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope('sent_LSTM'):
                sent_output, sent_state = self.sent_LSTM(project_vec, sent_state)

            with tf.name_scope('fc1'):
                hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
            with tf.name_scope('fc2'):
                sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

            # sent_state is a tuple, sent_state = (c, h)
            # 'c': shape=(1, 512) dtype=float32, 'h': shape=(1, 512) dtype=float32
            # The loss here, I refer from the web which is very helpful for me:
            # 1. http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
            # 2. http://stackoverflow.com/questions/35277898/tensorflow-for-binary-classification
            # 3. http://stackoverflow.com/questions/35226198/is-this-one-hot-encoding-in-tensorflow-fast-or-flawed-for-any-reason
            # 4. http://stackoverflow.com/questions/35198528/reshape-y-train-for-binary-text-classification-in-tensorflow
            sentRNN_logistic_mu = tf.nn.xw_plus_b( sent_output, self.logistic_Theta_W, self.logistic_Theta_b )
            sentRNN_label = tf.pack([ 1 - num_distribution[:, i], num_distribution[:, i] ])
            sentRNN_label = tf.transpose(sentRNN_label)
            sentRNN_loss = tf.nn.softmax_cross_entropy_with_logits(sentRNN_logistic_mu, sentRNN_label)
            sentRNN_loss = tf.reduce_sum(sentRNN_loss)/self.batch_size
            loss += sentRNN_loss * lambda_sent
            loss_sent += sentRNN_loss

            # the begining input of word_LSTM is topic vector, and DON'T compute the loss
            # This is follow the paper: Show and Tell
            #word_state = self.word_LSTM.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            #with tf.variable_scope('word_LSTM'):
            #    word_output, word_state = self.word_LSTM(sent_topic_vec)
            topic = tf.nn.rnn_cell.LSTMStateTuple(sent_topic_vec[:, 0:512], sent_topic_vec[:, 512:])
            word_state = (topic, topic)
            for j in range(0, self.N_max):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, captions[:, i, j])

                with tf.variable_scope('word_LSTM'):
                    word_output, word_state = self.word_LSTM(current_embed, word_state)

                # How to make one-hot encoder, I refer from this excellent web:
                # http://stackoverflow.com/questions/33681517/tensorflow-one-hot-encoder
                labels = tf.reshape(captions[:, i, j+1], [-1, 1])
                indices = tf.reshape(tf.range(0, self.batch_size, 1), [-1, 1])
                concated = tf.concat(1, [indices, labels])
                onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

                # At each timestep the hidden state of the last LSTM layer is used to predict a distribution
                # over the words in the vocbulary
                logit_words = tf.nn.xw_plus_b(word_output[:], self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                cross_entropy = cross_entropy * captions_masks[:, i, j]
                loss_wordRNN = tf.reduce_sum(cross_entropy) / self.batch_size
                loss += loss_wordRNN * lambda_word
                loss_word += loss_wordRNN

        return feats, num_distribution, captions, captions_masks, loss, loss_sent, loss_word

    def generate_model(self):
        # feats: 1 x 50 x 4096
        feats = tf.placeholder(tf.float32, [1, self.num_boxes, self.feats_dim])
        # tmp_feats: 50 x 4096
        tmp_feats = tf.reshape(feats, [-1, self.feats_dim])

        # project_vec_all: 50 x 4096 * 4096 x 1024 + 1024 --> 50 x 1024
        project_vec_all = tf.matmul(tmp_feats, self.regionPooling_W) + self.regionPooling_b
        project_vec_all = tf.reshape(project_vec_all, [1, 50, self.project_dim])
        project_vec = tf.reduce_max(project_vec_all, reduction_indices=1)

        # initialize the sent_LSTM state
        sent_state = self.sent_LSTM.zero_state(batch_size=1, dtype=tf.float32)

        # save the generated paragraph to list, here I named generated_sents
        generated_paragraph = []

        # pred
        pred_re = []

        # T_stop: run the sentence RNN forward until the stopping probability p_i (STOP) exceeds a threshold T_stop
        T_stop = tf.constant(0.5)

        # Start build the generation model
        print 'Start build the generation model: '

        # sentence RNN
        #word_state = self.word_LSTM.zero_state(batch_size=1, dtype=tf.float32)
        #with tf.variable_scope('word_LSTM'):
        #    word_output, word_state = self.word_LSTM(sent_topic_vec, word_state)
        for i in range(0, self.S_max):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # sent_state:
            # LSTMStateTuple(c=<tf.Tensor 'sent_LSTM/BasicLSTMCell/add_2:0' shape=(1, 512) dtype=float32>,
            #                h=<tf.Tensor 'sent_LSTM/BasicLSTMCell/mul_2:0' shape=(1, 512) dtype=float32>)
            with tf.variable_scope('sent_LSTM'):
                sent_output, sent_state = self.sent_LSTM(project_vec, sent_state)

            # self.fc1_W: 512 x 1024, self.fc1_b: 1024
            # hidden1: 1 x 1024
            # sent_topic_vec: 1 x 1024
            with tf.name_scope('fc1'):
                hidden1 = tf.nn.relu( tf.matmul(sent_output, self.fc1_W) + self.fc1_b )
            with tf.name_scope('fc2'):
                sent_topic_vec = tf.nn.relu( tf.matmul(hidden1, self.fc2_W) + self.fc2_b )

            sentRNN_logistic_mu = tf.nn.xw_plus_b(sent_output, self.logistic_Theta_W, self.logistic_Theta_b)
            pred = tf.nn.softmax(sentRNN_logistic_mu)
            pred_re.append(pred)

            # save the generated sentence to list, named generated_sent
            generated_sent = []

            # initialize the word LSTM state
            #word_state = self.word_LSTM.zero_state(batch_size=1, dtype=tf.float32)
            #with tf.variable_scope('word_LSTM'):
            #    word_output, word_state = self.word_LSTM(sent_topic_vec, word_state)
            topic = tf.nn.rnn_cell.LSTMStateTuple(sent_topic_vec[:, 0:512], sent_topic_vec[:, 512:])
            word_state = (topic, topic)
            # word RNN, unrolled to N_max time steps
            for j in range(0, self.N_max):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()

                if j == 0:
		    with tf.device('/cpu:0'):
			# get word embedding of BOS (index = 0)
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.zeros([1], dtype=tf.int64))

                with tf.variable_scope('word_LSTM'):
                    word_output, word_state = self.word_LSTM(current_embed, word_state)

                # word_state:
                # (
                #     LSTMStateTuple(c=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell0/BasicLSTMCell/add_2:0' shape=(1, 512) dtype=float32>,
                #                    h=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell0/BasicLSTMCell/mul_2:0' shape=(1, 512) dtype=float32>),
                #     LSTMStateTuple(c=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell1/BasicLSTMCell/add_2:0' shape=(1, 512) dtype=float32>,
                #                    h=<tf.Tensor 'word_LSTM_152/MultiRNNCell/Cell1/BasicLSTMCell/mul_2:0' shape=(1, 512) dtype=float32>)
                # )
                logit_words = tf.nn.xw_plus_b(word_output, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_sent.append(max_prob_index)

                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                    current_embed = tf.expand_dims(current_embed, 0)

            generated_paragraph.append(generated_sent)

        return feats, generated_paragraph, pred_re


# -----------------------------------------------------------------------------------------------------
# Preparing Functions
# -----------------------------------------------------------------------------------------------------
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )

    word_counts = {}
    nsents = 0

    for sent in sentence_iterator:
        nsents += 1
        tmp_sent = sent.lower().split(' ')
        if '' in tmp_sent:
            tmp_sent.remove('')

        for w in tmp_sent:
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '<bos>'
    ixtoword[1] = '<eos>'
    ixtoword[2] = '<pad>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<bos>'] = 0
    wordtoix['<eos>'] = 1
    wordtoix['<pad>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx + 4
        ixtoword[idx+4] = w

    word_counts['<eos>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<pad>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector


#######################################################################################################
# Parameters Setting
#######################################################################################################
batch_size = 50 # Being support batch_size
num_boxes = 50 # number of Detected regions in each image
feats_dim = 4096 # feature dimensions of each regions
project_dim = 1024 # project the features to one vector, which is 1024 dimensions

sentRNN_lstm_dim = 512 # the sentence LSTM hidden units
sentRNN_FC_dim = 1024 # the fully connected units
wordRNN_lstm_dim = 512 # the word LSTM hidden units
word_embed_dim = 1024 # the learned embedding vectors for the words

S_max = 6
N_max = 30
T_stop = 0.5

n_epochs = 500
learning_rate = 0.0001


#######################################################################################################
# Word vocubulary and captions preprocessing stage
#######################################################################################################
img2paragraph = pickle.load(open('./data/img2paragraph', 'rb'))
all_sentences = []
for key, paragraph in img2paragraph.iteritems():
    for each_sent in paragraph[1]:
        each_sent.replace(',', ' ,')
        all_sentences.append(each_sent)
word2idx, idx2word, bias_init_vector = preProBuildWordVocab(all_sentences, word_count_threshold=2)
np.save('./data/idx2word_batch', idx2word)

img2paragraph_modify = {}
for img_name, img_paragraph in img2paragraph.iteritems():
    img_paragraph_1 = img_paragraph[1]

    # img_paragraph_1 is a list
    # it may contain the element: '' or ' ', like this:
    # [["a man is walking"], ["the dog is running"], [""], [" "]]
    # so, we should remove them ' ' and '' element
    if '' in img_paragraph_1:
        img_paragraph_1.remove('')
    if ' ' in paragraph[1]:
        img_paragraph_1.remove(' ')

    # the number sents in each paragraph
    # if the sents is bigger than S_max,
    # we force the number of sents to be S_max
    img_num_sents = len(img_paragraph_1)
    if img_num_sents > S_max:
        img_num_sents = S_max

    # if a paragraph has 4 sentences
    # then the img_num_distribution will be like this:
    # [0, 0, 0, 1, 1, 1]
    img_num_distribution = np.zeros([S_max], dtype=np.int32)
    img_num_distribution[img_num_sents-1:] = 1

    # we multiply the number 2, because the <pad> is encoded into 2
    img_captions_matrix = np.ones([S_max, N_max+1], dtype=np.int32) * 2 # zeros([6, 50])
    for idx, img_sent in enumerate(img_paragraph_1):
        # the number of sentences is img_num_sents
        if idx == img_num_sents:
            break

        # because we treat the ',' as a word
        img_sent = img_sent.replace(',', ' ,')

        # Because I have preprocess the paragraph_v1.json file in VScode before,
        # and I delete all the 2, 3, 4...bankspaces
        # so, actually, the 'elif' code will never run
        if img_sent[0] == ' ' and img_sent[1] != ' ':
            img_sent = img_sent[1:]
        elif img_sent[0] == ' ' and img_sent[1] == ' ' and img_sent[2] != ' ':
            img_sent = img_sent[2:]

        # Be careful the last part in a sentence, like this:
        # '...world.'
        # '...world. '
        if img_sent[-1] == '.':
            img_sent = img_sent[0:-1]
        elif img_sent[-1] == ' ' and img_sent[-2] == '.':
            img_sent = img_sent[0:-2]

        # Last, we add the <bos> and the <eos> in each sentences
        img_sent = '<bos> ' + img_sent + ' <eos>'

        # translate each word in a sentence into the unique number in word2idx dict
        # when we meet the word which is not in the word2idx dict, we use the mark: <unk>
        for idy, word in enumerate(img_sent.lower().split(' ')):
            # because the biggest number of words in a sentence is N_max, here is 50
            if idy == N_max:
                break

            if word in word2idx:
                img_captions_matrix[idx, idy] = word2idx[word]
            else:
                img_captions_matrix[idx, idy] = word2idx['<unk>']

    # Pay attention, the value type 'img_name' here is NUMBER, I change it to STRING type
    img2paragraph_modify[str(img_name)] = [img_num_distribution, img_captions_matrix]

with open('./data/img2paragraph_modify_batch', 'wb') as f:
    pickle.dump(img2paragraph_modify, f)


#######################################################################################################
# Train, validation and testing stage
#######################################################################################################
def train():
    ##############################################################################
    # some preparing work
    ##############################################################################
    model_path = './models_batch/'
    train_feats_path = './data/im2p_train_output.h5'
    train_output_file = h5py.File(train_feats_path, 'r')
    train_feats = train_output_file.get('feats')
    train_imgs_full_path_lists = open('./densecap/imgs_train_path.txt').read().splitlines()
    train_imgs_names = map(lambda x: os.path.basename(x).split('.')[0], train_imgs_full_path_lists)


    # Model Initialization:
    # n_words, batch_size, num_boxes, feats_dim, project_dim, sentRNN_lstm_dim, sentRNN_FC_dim, wordRNN_lstm_dim, S_max, N_max
    model = RegionPooling_HierarchicalRNN(n_words = len(word2idx),
                                          batch_size = batch_size,
                                          num_boxes = num_boxes,
                                          feats_dim = feats_dim,
                                          project_dim = project_dim,
                                          sentRNN_lstm_dim = sentRNN_lstm_dim,
                                          sentRNN_FC_dim = sentRNN_FC_dim,
                                          wordRNN_lstm_dim = wordRNN_lstm_dim,
                                          S_max = S_max,
                                          N_max = N_max,
                                          word_embed_dim = word_embed_dim,
                                          bias_init_vector = bias_init_vector)

    tf_feats, tf_num_distribution, tf_captions_matrix, tf_captions_masks, tf_loss, tf_loss_sent, tf_loss_word = model.build_model()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver(max_to_keep=500, write_version=1)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()

    # when you want to train the model from the front model
    #new_saver = tf.train.Saver(max_to_keep=500)
    #new_saver = tf.train.import_meta_graph('./models_batch/model-92.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./models_batch/'))

    all_vars = tf.trainable_variables()

    # open a loss file to record the loss value
    loss_fd = open('loss_batch.txt', 'w')
    img2idx = {}
    for idx, img in enumerate(train_imgs_names):
        img2idx[img] = idx

    # plt draw the loss curve
    # refer from: http://stackoverflow.com/questions/11874767/real-time-plotting-in-while-loop-with-matplotlib
    loss_to_draw = []

    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []
        # disorganize the order
        random.shuffle(train_imgs_names)

        for start, end in zip(range(0, len(train_imgs_names), batch_size),
                              range(batch_size, len(train_imgs_names), batch_size)):

            start_time = time.time()

            img_name = train_imgs_names[start:end]
            current_feats_index = map(lambda x: img2idx[x], img_name)
            current_feats = np.asarray( map(lambda x: train_feats[x], current_feats_index) )

            current_num_distribution = np.asarray( map(lambda x: img2paragraph_modify[x][0], img_name) )
            current_captions_matrix = np.asarray( map(lambda x: img2paragraph_modify[x][1], img_name) )

            current_captions_masks = np.zeros( (current_captions_matrix.shape[0], current_captions_matrix.shape[1], current_captions_matrix.shape[2]) )
            # find the non-zero element
            nonzeros = np.array( map(lambda each_matrix: np.array( map(lambda x: (x != 2).sum() + 1, each_matrix ) ), current_captions_matrix ) )
            for i in range(batch_size):
                for ind, row in enumerate(current_captions_masks[i]):
                    row[:(nonzeros[i, ind]-1)] = 1

            # shape of current_feats: batch_size x 50 x 4096
            # shape of current_num_distribution: batch_size x 6
            # shape of current_captions_matrix: batch_size x 6 x 50
            _, loss_val, loss_sent, loss_word= sess.run(
                                [train_op, tf_loss, tf_loss_sent, tf_loss_word],
                                feed_dict={
                                           tf_feats: current_feats,
                                           tf_num_distribution: current_num_distribution,
                                           tf_captions_matrix: current_captions_matrix,
                                           tf_captions_masks: current_captions_masks
                                })

            # append loss to list in a epoch
            loss_to_draw_epoch.append(loss_val)

            # running information
            print 'idx: ', start, ' Epoch: ', epoch, ' loss: ', loss_val, ' loss_sent: ', loss_sent, ' loss_word: ', loss_word, ' Time cost: ', str((time.time() - start_time))
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val))

        # draw loss curve every epoch
        loss_to_draw.append(np.mean(loss_to_draw_epoch))
        plt_save_dir = './loss_imgs'
        plt_save_img_name = str(epoch) + '.png'
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)
        plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

        if np.mod(epoch, 10) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
    loss_fd.close()


def test():
    start_time = time.time()
    # change the model path according to your environment
    model_path = './models_batch/model-500'

    # It's very important to use Pandas to Series this idx2word dict
    # After this operation, we can use list to extract the word at the same time
    idx2word = pd.Series(np.load('./data/idx2word_batch.npy').tolist())

    test_feats_path = './data/im2p_test_output.h5'
    test_output_file = h5py.File(test_feats_path, 'r')
    test_feats = test_output_file.get('feats')

    test_imgs_full_path_lists = open('./densecap/imgs_test_order.txt').read().splitlines()
    test_imgs_names = map(lambda x: os.path.basename(x).split('.')[0], test_imgs_full_path_lists)

    # n_words, batch_size, num_boxes, feats_dim, project_dim, sentRNN_lstm_dim, sentRNN_FC_dim, wordRNN_lstm_dim, S_max, N_max
    test_model = RegionPooling_HierarchicalRNN(n_words = len(word2idx),
                                               batch_size = batch_size,
                                               num_boxes = num_boxes,
                                               feats_dim = feats_dim,
                                               project_dim = project_dim,
                                               sentRNN_lstm_dim = sentRNN_lstm_dim,
                                               sentRNN_FC_dim = sentRNN_FC_dim,
                                               wordRNN_lstm_dim = wordRNN_lstm_dim,
                                               S_max = S_max,
                                               N_max = N_max,
                                               word_embed_dim = word_embed_dim,
                                               bias_init_vector = bias_init_vector)

    tf_feats, tf_generated_paragraph, tf_pred_re, tf_sent_topic_vectors = test_model.generate_model()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    img2idx = {}
    for idx, img in enumerate(test_imgs_names):
        img2idx[img] = idx

    test_fd = open('HRNN_results.txt', 'w')
    for idx, img_name in enumerate(test_imgs_names):
        print idx, img_name
        test_fd.write(img_name + '\n')

        each_paragraph = []
        current_paragraph = ""

        current_feats_index = img2idx[img_name]
        current_feats = test_feats[current_feats_index]
        current_feats = np.reshape(current_feats, [1, 50, 4096])

        generated_paragraph_indexes, pred, sent_topic_vectors = sess.run(
                                                                         [tf_generated_paragraph, tf_pred_re, tf_sent_topic_vectors],
                                                                         feed_dict={
                                                                             tf_feats: current_feats
                                                                         })

        #generated_paragraph = idx2word[generated_paragraph_indexes]
        for sent_index in generated_paragraph_indexes:
            each_sent = []
            for word_index in sent_index:
                each_sent.append(idx2word[word_index])
            each_paragraph.append(each_sent)

        for idx, each_sent in enumerate(each_paragraph):
            # if the current sentence is the end sentence of the paragraph
            # According to the probability distribution:
            # CONTINUE: [1, 0]
            # STOP    : [0, 1]
            # So, if the first item of pred is less than the T_stop
            # the generation process is break
            if pred[idx][0][0] <= T_stop:
                break
            current_sent = ''
            for each_word in each_sent:
                current_sent += each_word + ' '
            current_sent = current_sent.replace('<eos> ', '')
            current_sent = current_sent.replace('<pad> ', '')
            current_sent = current_sent + '.'
            current_sent = current_sent.replace(' .', '.')
            current_sent = current_sent.replace(' ,', ',')
            current_paragraph +=current_sent
            if idx != len(each_paragraph) - 1:
                current_paragraph += ' '

        test_fd.write(current_paragraph + '\n')
    test_fd.close()
    print "Time cost: " + str(time.time()-start_time)


