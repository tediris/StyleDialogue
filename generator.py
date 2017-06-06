import tensorflow as tf


class Generator:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.FLAGS = FLAGS
        self.setup_placeholders()
        self.setup_embeddings()
        self.encoder()
        self.decoder()
        self.loss_op()

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
        hidden_size = 500
        cell_fw = tf.contrib.rnn.GRUCell(hidden_size)
        cell_bw = tf.contrib.rnn.GRUCell(hidden_size)
        outputs, output_states = bidirectional_dynamic_rnn(
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
        hidden_dim = 1000
        cell = tf.contrib.rnn.GRUCell(hidden_dim)
        # outputs, final_state = tf.nn.dynamic_rnn(cell, self.encoder_out, sequence_length=self.lines_len_placeholder, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.encoder_out, dtype=tf.float32)
        # outputs is batch_size x max_time x 100
        max_time = tf.shape(outputs)[1] # this should be dynamic
        batch_size = tf.shape(outputs)[0]
        reshaped = tf.reshape(outputs, [batch_size * max_time, hidden_dim])
        # classify all of these as words
        self.logits = tf.layers.dense(reshaped, self.vocab_size) # (batch_size x max_time) x vocab
        decoded_words = tf.argmax(preds, axis=1) # (batch_size x max_time)
        self.decoded_sentences = tf.reshape(decoded_words, [batch_size, max_time])

    def loss_op(self):
        batch_size = tf.shape(self.lines_placeholder)[0]
        max_time = tf.shape(self.lines_placeholder)[1]
        reshaped_input = tf.reshape(self.lines_placeholder, [batch_size * max_time,])
        one_hot_labels = tf.one_hot(reshaped_input, depth=self.vocab_size) # (batch_size x max_time) x vocab
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=self.logits))

    def optimization_op(self):
        optimizer = tf.train.AdamOptimizer() # select optimizer and set learning rate
        # batch normalization in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_step = optimizer.minimize(self.loss)
