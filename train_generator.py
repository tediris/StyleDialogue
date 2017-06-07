from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random

import tensorflow as tf

# from qa_model import Encoder, QASystem, Decoder
# from model import Classifier
from generator import Generator
from data import PAD_ID
from os.path import join as pjoin
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "dialog_corpus/movie", "SQuAD directory (default ./dialog_corpus/movie)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


# def initialize_model(session, model, train_dir):
#     ckpt = tf.train.get_checkpoint_state(train_dir)
#     v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
#     if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
#         logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
#         model.saver.restore(session, ckpt.model_checkpoint_path)
#     else:
#         logging.info("Created model with fresh parameters.")
#         session.run(tf.global_variables_initializer())
#         logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
#     return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with open(vocab_path, 'r') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    def read_from_file(filename):
        f = open(filename)
        lines = f.readlines()
        f.close()
        #convert list of strings to list of list of ints
        # datalist = [map(int, str.split(line)) for line in lines]
        datalist = [[int(word) for word in line.split()] for line in lines]
        return datalist

    def read_labels_from_file(filename):
        f = open(filename)
        lines = f.readlines()
        f.close()
        #convert list of strings to list of list of ints
        datalist = [int(line) for line in lines]
        return datalist

    def maxLen(arrayOfArrays):
        return max([len(array) for array in arrayOfArrays])

    def padded(array, length):
        return array + [str(PAD_ID)] * (length - len(array))

    '''returns a dictionary mapping filename to int id'''
    def filesIn(directory, endsWith):
        allFiles = os.listdir(directory)
        filteredFiles = {}
        for f in allFiles:
            if f.endswith(endsWith):
                filteredFiles[directory + "/" + f] = len(filteredFiles)
        return filteredFiles

    def appendFiles(files, minLength=0):
        allLines = []
        allIDs = []
        for filename in files:
            fileID = files[filename]
            _open = open(filename)
            lines = _open.readlines()
            filteredLines = []
            for line in lines:
                if len(line) > minLength:
                    filteredLines.append(line.split())
            lines = filteredLines
            allLines = allLines + lines
            allIDs = allIDs + [fileID]*len(lines)
        return allLines, allIDs

    '''data is of the form list of list of sentences (list of ints)'''
    def printToFile(filename, data):
        _open = open(filename, 'w')
        for listOfSentences in data:
            for sentence in listOfSentences:
                _open.write(" ".join([str(x) for x in sentence]) + "\n")

    '''I am so sorry. This one writes list of ints or list of lists'''
    def printToFile_standard(filename, data):
        _open = open(filename, 'w')
        for line in data:
            if type(line) is int:
                _open.write(str(line) + "\n")
            else:
                _open.write(" ".join([str(x) for x in line]) + "\n")

    # novels = filesIn('data', '.txt.ids')
    # allLines, allIDs = appendFiles(novels, 5)
    # # maxLen = maxLen(allLines)
    # # allLines = [padded(line, maxLen) for line in allLines]
    # # print(allLines[:10])
    # printToFile("novel_ids.txt", allIDs)
    # printToFile("novel_lines.txt", allLines)
    lines = read_from_file("novel_lines.txt")
    ids = read_labels_from_file("novel_ids.txt")
    index_shuf = list(range(len(lines)))
    random.shuffle(index_shuf)
    lines = [lines[i] for i in index_shuf]
    printToFile_standard('shuffled_input.txt', lines)
    ids = [ids[i] for i in index_shuf]
    style_vector = np.load('data/jk_rowling_mean.npy')

    dataset = (lines, style_vector)
    #
    embed_path = "glove.trimmed.{}.npz".format(FLAGS.embedding_size)
    vocab_path = pjoin("vocab", "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    model = Generator(FLAGS, len(vocab))
    #
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    #
    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)
    #
    with tf.Session() as sess:
        # load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        # initialize_model(sess, qa, load_train_dir)
        classifier_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="sentence")
        # print(classifier_weights)
        classifier_saver = tf.train.Saver(classifier_weights)

        # save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        save_train_dir = "None"
        model.initialize(sess)
        # classifier_saver.restore(sess, 'ckpts/sentence_classifier/model.ckpt-1')
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, 'ckpts/generator/model.ckpt-1')

        model.train(sess, dataset, save_train_dir)
        # decoded_ids = model.generate(sess, dataset, save_train_dir)
        # print(type(decoded_ids[0][0]))
        # printToFile("decoded_ids.txt", decoded_ids)
        saver.save(sess, 'ckpts/generator/model.ckpt', global_step=1)

        # qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
