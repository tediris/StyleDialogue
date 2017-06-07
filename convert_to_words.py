from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random

import tensorflow as tf

# from qa_model import Encoder, QASystem, Decoder
from model import Classifier
from data import PAD_ID
from os.path import join as pjoin
import numpy as np

import logging

TRAINING = False

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use during training.")
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
        print(rev_vocab[:10])
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


# def get_normalized_train_dir(train_dir):
#     """
#     Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
#     file paths saved in the checkpoint. This allows the model to be reloaded even
#     if the location of the checkpoint files has moved, allowing usage with CodaLab.
#     This must be done on both train.py and qa_answer.py in order to work.
#     """
#     global_train_dir = '/tmp/cs224n-squad-train'
#     if os.path.exists(global_train_dir):
#         os.unlink(global_train_dir)
#     if not os.path.exists(train_dir):
#         os.makedirs(train_dir)
#     os.symlink(os.path.abspath(train_dir), global_train_dir)
#     return global_train_dir

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

'''accepts list of ints or list of lists'''
def printToFile(filename, data):
    _open = open(filename, 'w')
    for line in data:
        if type(line) is int:
            _open.write(str(line) + "\n")
        else:
            _open.write(" ".join(line) + "\n")

def idsToWords(listOfSentences, rev_vocab):
    words = []
    print(listOfSentences[0])
    for sentence in listOfSentences:
        # sentence = sentence.split()
        wordsInSentence = [rev_vocab[x] for x in sentence]
        filters = ['<pad>']
        filteredWords = []
        for word in wordsInSentence:
            if word not in filters:
                filteredWords.append(word)
        words.append(filteredWords)
    return words

# dataset = (lines, ids)
#
embed_path = "glove.trimmed.{}.npz".format(FLAGS.embedding_size)
vocab_path = pjoin("vocab", "vocab.dat")
vocab, rev_vocab = initialize_vocab(vocab_path)

ids = read_from_file('decoded_ids.txt')
words = idsToWords(ids, rev_vocab)

printToFile('decoded_words.txt', words)

ids = read_from_file('shuffled_input.txt')
words = idsToWords(ids, rev_vocab)

printToFile('shuffled_input_words.txt', words)
print("DONE")
