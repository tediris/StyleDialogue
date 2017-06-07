import tensorflow as tf
import numpy as np
from data import PAD_ID
from tensorflow.python.ops import variable_scope as vs
import time
import logging


class Generator:
    def __init__(self, FLAGS, vocab_size):
        self.vocab_size = vocab_size
        self.FLAGS = FLAGS
        self.setup_placeholders()
        self.setup_embeddings()
        self.encoder()
        self.decoder()
        self.loss_op()
        self.optimization_op()
        self.init_op = tf.global_variables_initializer()

    def initialize(self, session):
        session.run(self.init_op)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding_files = np.load("glove.trimmed.100.npz")
            self.pretrained_embeddings = tf.constant(embedding_files["glove"], dtype=tf.float32)
            self.lines = tf.nn.embedding_lookup(self.pretrained_embeddings, self.lines_placeholder)

    def setup_placeholders(self):
        self.lines_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="lines_place")
        self.lines_len_placeholder = tf.placeholder(tf.int32, shape=[None], name="lines_mask")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None], name="labels_place")

    def encoder(self):
        hidden_size = 100
        cell_fw = tf.contrib.rnn.GRUCell(hidden_size)
        cell_bw = tf.contrib.rnn.GRUCell(hidden_size)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            self.lines,
            sequence_length=self.lines_len_placeholder,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype=tf.float32,
        )

        self.encoder_out = tf.concat([outputs[0], outputs[1]], axis=2)

    def decoder(self):
        # input is 1000 dimensional
        hidden_dim = 200
        cell = tf.contrib.rnn.GRUCell(hidden_dim)
        # outputs, final_state = tf.nn.dynamic_rnn(cell, self.encoder_out, sequence_length=self.lines_len_placeholder, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.encoder_out, dtype=tf.float32)
        # outputs is batch_size x max_time x 100
        max_time = tf.shape(outputs)[1] # this should be dynamic
        batch_size = tf.shape(outputs)[0]
        reshaped = tf.reshape(outputs, [batch_size * max_time, hidden_dim])
        # classify all of these as words
        self.logits = tf.layers.dense(reshaped, self.vocab_size) # (batch_size x max_time) x vocab
        decoded_words = tf.argmax(self.logits, axis=1) # (batch_size x max_time)
        self.decoded_sentences = tf.reshape(decoded_words, [batch_size, max_time])

    def loss_op(self):
        batch_size = tf.shape(self.lines_placeholder)[0]
        max_time = tf.shape(self.lines_placeholder)[1]
        reshaped_input = tf.reshape(self.lines_placeholder, [batch_size * max_time,])

        # self.lines are the looked-up input, batch_size x max_time x 100
        # pretrained_embeddings is vocab_size x 100
        lines_flat = tf.reshape(self.lines, [batch_size * max_time, -1])
        lines_flat = tf.nn.l2_normalize(lines_flat, dim=-1)
        embeddings_norm = tf.nn.l2_normalize(self.pretrained_embeddings, dim=-1)
        vocab_weights = tf.matmul(lines_flat, tf.transpose(embeddings_norm))

        ### TESTING THIS ###
        probs = tf.nn.softmax(self.logits, dim=-1)

        # self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=self.logits))
        self.loss = tf.reduce_mean(tf.losses.cosine_distance(vocab_weights, probs, dim=-1))
        # this is (batch_size x time) x vocab_size
        # vocab_weights = tf.nn.l2_normalize(vocab_weights, dim=-1)
        # vocab_weights = 1.0 - (vocab_weights / tf.reduce_max(vocab_weights, axis=-1))
        # probs = tf.nn.softmax(self.logits, dim=-1)
        # probs = tf.nn.log_softmax(self.logits, dim=-1)
        # weighted_scores = vocab_weights * probs
        # self.loss = -tf.reduce_mean(weighted_scores)
        # one_hot_labels = tf.one_hot(reshaped_input, depth=self.vocab_size) # (batch_size x max_time) x vocab
        # self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=self.logits))

    def optimization_op(self):
        optimizer = tf.train.AdamOptimizer() # select optimizer and set learning rate
        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def make_batch(self, dataset, iteration):
        batch_size = self.FLAGS.batch_size
        train_lines = dataset
        start_index = iteration*batch_size

        #make padded enc batch
        lines = train_lines[start_index:start_index+batch_size]
        lines_len = np.array([len(q) for q in lines])
        lines_max_len = np.max(lines_len)
        lines_batch = np.array([q + [PAD_ID]*(lines_max_len - len(q)) for q in lines])

        return lines_batch, lines_len

    # def build_model(self):
    #     self.test =
    def optimize(self, session, data):
        lines_batch, lines_len = data
        feed_dict = {}
        feed_dict[self.lines_placeholder] = lines_batch
        feed_dict[self.lines_len_placeholder] = lines_len

        output_feed = [self.loss, self.train_step]
        return session.run(output_feed, feed_dict)

    def decode(self, session, data):
        lines_batch, lines_len = data
        feed_dict = {}
        feed_dict[self.lines_placeholder] = lines_batch
        feed_dict[self.lines_len_placeholder] = lines_len

        output_feed = [self.loss, self.decoded_sentences]
        return session.run(output_feed, feed_dict)

    def train(self, session, dataset, train_dir):

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))


        #run main training loop: (only 1 epoch for now)
        train_lines = dataset
        max_iters = np.ceil(len(train_lines)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        for epoch in range(1):
            #temp hack to only train on some small subset:
            # max_iters = 1
            for iteration in range(int(max_iters)):
                print("Current iteration: " + str(iteration))
                lines_batch, lines_len = self.make_batch(dataset, iteration)
                # lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
                #retrieve useful info from training - see optimize() function to set what we're tracking
                loss, _ = self.optimize(session, (lines_batch, lines_len))
                # print("accuracy: " + str(accuracy))
                print("Current Loss: " + str(loss))
                # print(grad_norm)
