import tensorflow as tf
import numpy as np
from data import PAD_ID
from tensorflow.python.ops import variable_scope as vs
import time
import logging

NUM_CLASSES = 5


class Classifier:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.setup_placeholders()
        self.setup_embeddings()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_optimization_op()
        self.init_op = tf.global_variables_initializer()

    def initialize(self, session):
        session.run(self.init_op)

    def setup_placeholders(self):
        self.lines_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="lines_place")
        self.lines_len_placeholder = tf.placeholder(tf.int32, shape=[None], name="lines_mask")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None], name="labels_place")

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding_files = np.load("glove.trimmed.100.npz")
            self.pretrained_embeddings = tf.constant(embedding_files["glove"], dtype=tf.float32)
            self.lines = tf.nn.embedding_lookup(self.pretrained_embeddings, self.lines_placeholder)
            # self.decs = tf.nn.embedding_lookup(self.pretrained_embeddings, self.decs_placeholder)

    # def encode(self):
    #     hidden_size = 100
    #     self.test_size = tf.shape(self.encs)
    #     cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    #     outputs, final_state = tf.nn.dynamic_rnn(cell, self.encs, sequence_length=self.encs_len_placeholder, dtype=tf.float32)
    #     self.encode_output = outputs
    #     self.encode_thought = final_state

    def add_prediction_op(self):
        hidden_size = 100
        self.test_size = tf.shape(self.lines)
        with vs.variable_scope("sentence"):
            cell = tf.contrib.rnn.GRUCell(hidden_size)
            outputs, final_state = tf.nn.dynamic_rnn(cell, self.lines, sequence_length=self.lines_len_placeholder, dtype=tf.float32)
            self.features = final_state
            self.mean_features = tf.reduce_mean(self.features, axis=0)
            self.logits = tf.layers.dense(final_state, 5)

    def add_loss_op(self):
        one_hot_labels = tf.one_hot(self.labels_placeholder, depth=NUM_CLASSES)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=self.logits))
        labels = tf.cast(self.labels_placeholder, tf.int64)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def add_optimization_op(self):
        optimizer = tf.train.AdamOptimizer() # select optimizer and set learning rate
        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_step = optimizer.minimize(self.loss)

    def make_batch(self, dataset, iteration):
        batch_size = self.FLAGS.batch_size
        train_lines, train_labels = dataset
        start_index = iteration*batch_size

        #make padded enc batch
        lines = train_lines[start_index:start_index+batch_size]
        lines_len = np.array([len(q) for q in lines])
        lines_max_len = np.max(lines_len)
        lines_batch = np.array([q + [PAD_ID]*(lines_max_len - len(q)) for q in lines])

        labels_batch = train_labels[start_index:start_index+batch_size]


        return lines_batch, lines_len, labels_batch

    # def build_model(self):
    #     self.test =
    def optimize(self, session, data):
        lines_batch, lines_len, labels_batch = data
        feed_dict = {}
        feed_dict[self.lines_placeholder] = lines_batch
        feed_dict[self.lines_len_placeholder] = lines_len
        feed_dict[self.labels_placeholder] = labels_batch

        output_feed = [self.loss, self.accuracy, self.train_step]
        return session.run(output_feed, feed_dict)

    def classify(self, session, data):
        lines_batch, lines_len, labels_batch = data
        feed_dict = {}
        feed_dict[self.lines_placeholder] = lines_batch
        feed_dict[self.lines_len_placeholder] = lines_len
        feed_dict[self.labels_placeholder] = labels_batch

        output_feed = [self.loss, self.accuracy]
        return session.run(output_feed, feed_dict)

    def generate_embeddings(self, session, data):
        lines_batch, lines_len, labels_batch = data
        feed_dict = {}
        feed_dict[self.lines_placeholder] = lines_batch
        feed_dict[self.lines_len_placeholder] = lines_len
        feed_dict[self.labels_placeholder] = labels_batch

        output_feed = [self.mean_features]
        return session.run(output_feed, feed_dict)

    def test(self, session, dataset):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))


        #run main training loop: (only 1 epoch for now)
        train_lines, train_labels = dataset
        max_iters = np.ceil(len(train_lines)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        accuracies = []
        for iteration in range(int(max_iters)):
            # print("Current iteration: " + str(iteration))
            if iteration % 10 == 0:
                print("Current iteration: " + str(iteration))

            lines_batch, lines_len, labels_batch = self.make_batch(dataset, iteration)
            # lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
            #retrieve useful info from training - see optimize() function to set what we're tracking
            loss, accuracy = self.classify(session, (lines_batch, lines_len, labels_batch))
            # print("accuracy: " + str(accuracy))
            accuracies.append(accuracy)
        accuracy = np.mean(np.array(accuracies))
        print("Total accuracy:" + str(accuracy))

    def generate_mean_embeddings(self, session, dataset):
        train_lines, train_labels = dataset
        max_iters = np.ceil(len(train_lines)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        mean_features = np.zeros(100)
        for iteration in range(int(max_iters)):
            # print("Current iteration: " + str(iteration))
            if iteration % 10 == 0:
                print("Current iteration: " + str(iteration))

            lines_batch, lines_len, labels_batch = self.make_batch(dataset, iteration)
            # lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
            #retrieve useful info from training - see optimize() function to set what we're tracking
            features = self.generate_embeddings(session, (lines_batch, lines_len, labels_batch))
            features = features[0]
            # print(len(features))
            # print("accuracy: " + str(accuracy))
            mean_features += np.array(features)
        mean_features /= max_iters
        # print("Total accuracy:" + str(accuracy))
        return mean_features


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
        train_lines, train_labels = dataset
        max_iters = np.ceil(len(train_lines)/float(self.FLAGS.batch_size))
        print("Max iterations: " + str(max_iters))
        for epoch in range(5):
            #temp hack to only train on some small subset:
            # max_iters = 1
            for iteration in range(int(max_iters)):
                lines_batch, lines_len, labels_batch = self.make_batch(dataset, iteration)
                # lr = tf.train.exponential_decay(self.FLAGS.learning_rate, iteration, 100, 0.96) #iteration here should be global when multiple epochs
                #retrieve useful info from training - see optimize() function to set what we're tracking
                loss, accuracy, _ = self.optimize(session, (lines_batch, lines_len, labels_batch))
                if iteration % 10 == 0:
                    print("Current iteration: " + str(iteration))
                    print("accuracy: " + str(accuracy))
                # print("Current Loss: " + str(loss))
                # print(grad_norm)
