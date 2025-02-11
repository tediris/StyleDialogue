import tensorflow as tf
import numpy as np
from data import PAD_ID
from tensorflow.python.ops import variable_scope as vs
import time
import logging


class Generator:
    def __init__(self, FLAGS, vocab_size, classifier=None):
        self.classifier = classifier
        self.vocab_size = vocab_size
        self.FLAGS = FLAGS
        self.setup_placeholders()
        self.setup_embeddings()
        self.encoder()
        self.decoder()
        self.add_style_op()
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
        # self.labels_placeholder = tf.placeholder(tf.int32, shape=[None], name="labels_place")
        self.style_placeholder = tf.placeholder(tf.int32, shape=[100], name="style_place")
        self.style_label_placeholder = tf.placeholder(tf.int32, shape=(), name="style_label_place")

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
        # input is 200 dimensional
        hidden_dim = 100 # TODO: this must match the glove dimensions
        cell = tf.contrib.rnn.GRUCell(hidden_dim)
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.encoder_out, sequence_length=self.lines_len_placeholder, dtype=tf.float32)
        # outputs, final_state = tf.nn.dynamic_rnn(cell, self.encoder_out, dtype=tf.float32)
        # outputs is batch_size x max_time x 100
        # self.decoder_output = tf.layers.dense(outputs, 100)
        max_time = tf.shape(outputs)[1] # this should be dynamic
        batch_size = tf.shape(outputs)[0]
        # classify all of these as words
        self.decode_output = outputs
        batch_size = tf.shape(self.lines_placeholder)[0]
        max_time = tf.shape(self.lines_placeholder)[1]
        # batch_size x max_time x 100

        # find the closest word embeddings
        decode_reshape = tf.reshape(outputs, [-1, 100])
        decode_normed = tf.nn.l2_normalize(decode_reshape, dim=1)
        embeddings_normed = tf.nn.l2_normalize(self.pretrained_embeddings, dim=1)

        cosine_similarity = tf.matmul(decode_normed, tf.transpose(embeddings_normed, [1, 0]))
        closest_words = tf.argmax(cosine_similarity, 1)
        self.decoded_ids = tf.reshape(closest_words, [batch_size, max_time])

    def add_style_op(self):
        hidden_size = 100
        with vs.variable_scope("sentence"):
            cell = tf.contrib.rnn.GRUCell(hidden_size)
            outputs, final_state = tf.nn.dynamic_rnn(cell, self.decode_output, sequence_length=self.lines_len_placeholder, dtype=tf.float32)
            # outputs, final_state = tf.nn.dynamic_rnn(cell, self.decode_output, sequence_length=None, dtype=tf.float32)
            # self.style_features = tf.stop_gradient(final_state)
            self.style_features = final_state
            # batch_size by style size
            # self.mean_features = tf.reduce_mean(self.features, axis=0)
            self.style_class = tf.layers.dense(final_state, 5)


    def loss_op(self):
        batch_size = tf.shape(self.lines_placeholder)[0]
        max_time = tf.shape(self.lines_placeholder)[1]

        def avg_pool_1d(input_layer, kernel=3, stride=2, padding='VALID'):
            input_reshape = tf.reshape(input_layer, [batch_size, max_time, 1, 100])
            pooled = tf.nn.avg_pool(input_reshape, ksize=[1, kernel, 1, 1], strides=[1, stride, 1, 1], padding=padding)
            return tf.reshape(pooled, [batch_size, -1, 100])

        lines_reduced = avg_pool_1d(self.lines)
        # outputs_reduced = self.decode_output#f.reduce_mean(self.decode_output, axis=1)
        outputs_reduced = avg_pool_1d(self.decode_output)
        self.content_loss = tf.losses.mean_squared_error(lines_reduced, outputs_reduced)
        # style_reshaped = tf.reshape(self.style_placeholder, [1, 100])
        # style_repeated = tf.tile(style_reshaped, [batch_size, 1])
        one_hot_labels = tf.one_hot(self.style_label_placeholder, depth=5)
        labels_reshaped = tf.reshape(one_hot_labels, [1, 5])
        labels_repeated = tf.tile(labels_reshaped, [batch_size, 1])
        # self.style_loss = tf.losses.mean_squared_error(style_repeated, self.style_features)
        self.style_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels_repeated, logits=self.style_class))
        alpha = 3.0
        beta = 1.0
        self.loss = alpha * self.content_loss + beta * self.style_loss

    def optimization_op(self):
        optimizer = tf.train.AdamOptimizer() # select optimizer and set learning rate
        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        vars_to_train = tf.trainable_variables()
        vars_to_ignore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "sentence")
        vars_to_train_filtered = [v for v in vars_to_train if v not in vars_to_ignore]
        # print(vars_to_train_filtered)
        with tf.control_dependencies(extra_update_ops):
            self.train_step = optimizer.minimize(self.loss, var_list=vars_to_train_filtered)

    def make_batch(self, dataset, iteration):
        batch_size = self.FLAGS.batch_size
        train_lines, _, style_label = dataset
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
        lines_batch, lines_len, style_vector, style_label = data
        feed_dict = {}
        feed_dict[self.lines_placeholder] = lines_batch
        feed_dict[self.lines_len_placeholder] = lines_len
        feed_dict[self.style_placeholder] = style_vector
        feed_dict[self.style_label_placeholder] = style_label

        output_feed = [self.loss, self.train_step]
        return session.run(output_feed, feed_dict)

    def decode(self, session, data):
        lines_batch, lines_len, style_vector, style_label = data
        feed_dict = {}
        feed_dict[self.lines_placeholder] = lines_batch
        feed_dict[self.lines_len_placeholder] = lines_len
        feed_dict[self.style_placeholder] = style_vector
        feed_dict[self.style_label_placeholder] = style_label

        output_feed = [self.loss, self.decoded_ids]
        return session.run(output_feed, feed_dict)

    def train(self, session, dataset, train_dir):

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))


        #run main training loop: (only 1 epoch for now)
        train_lines, style_vector, style_label = dataset
        max_iters = np.ceil(len(train_lines)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        for epoch in range(10):
            #temp hack to only train on some small subset:
            # max_iters = 1
            for iteration in range(int(max_iters)):
                print("Current iteration: " + str(iteration))
                lines_batch, lines_len = self.make_batch(dataset, iteration)
                # lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
                #retrieve useful info from training - see optimize() function to set what we're tracking
                loss, _ = self.optimize(session, (lines_batch, lines_len, style_vector, style_label))
                # print("accuracy: " + str(accuracy))
                print("Current Loss: " + str(loss))
                # print(grad_norm)

    def generate(self, session, dataset, train_dir):

        #run main training loop: (only 1 epoch for now)
        train_lines, style_vector = dataset
        max_iters = np.ceil(len(train_lines)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        # for epoch in range(1):
            #temp hack to only train on some small subset:
            # max_iters = 1
        all_decoded_ids = []
        for iteration in range(int(max_iters)):
            print("Current iteration: " + str(iteration))
            lines_batch, lines_len = self.make_batch(dataset, iteration)
            # lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
            #retrieve useful info from training - see optimize() function to set what we're tracking
            loss, decoded_ids = self.decode(session, (lines_batch, lines_len, style_vector, style_label))
            all_decoded_ids.append(decoded_ids)
            # print(decoded_ids)
            # print("accuracy: " + str(accuracy))
            print("Current Loss: " + str(loss))
            # print(grad_norm)
        return all_decoded_ids
