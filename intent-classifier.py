#!/usr/bin/env python3
"""
intent-classifier.py

Embedding Layer:
https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
https://github.com/CyberZHG/keras-bert

@Author: Fabian Fey, Gianna Weber
"""


import numpy as np
import logging
import logzero
import pickle
import os
from datetime import datetime

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from bert_embedding import BertEmbedding

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

DATA_SET_TYPE = "train"

MAX_SENTENCE_LENGTH = 30
EMBEDDING_SIZE = 800  # lenght of vector created by BERT

EPOCHS = 15
BATCH_SIZE = 128
ACTIVATION = ""
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
DROPOUT = 0.4
LEARNING_RATE = 0.001
TRAINABLE_EMBEDDINGS = True

logzero.logfile("Log/intent-logs.log", maxBytes=1e6, backupCount=5)
logzero.loglevel(10)
logger = logzero.logger

TensorBoardLog = "Log/tensorboard/intent/" + \
    datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(TensorBoardLog)


logger.info("----Start of Program----")
logger.debug("Data Set Type: %s", DATA_SET_TYPE)
logger.debug("Learning Rate: %s ", LEARNING_RATE)
logger.debug("Batch Size: %s ", BATCH_SIZE)
logger.debug("Epochs: %s ", EPOCHS)
logger.debug("Embedding Size %s ", EMBEDDING_SIZE)
logger.debug("Trainable Embedding: %s", TRAINABLE_EMBEDDINGS)


def main():

    bert_embedding = BertEmbedding()

    sentences_Train, sentences_Test, labels_Train, labels_Test, vocab_size, tokenizer = process_Data()

    logger.debug("vocab_size: %s", vocab_size)

    # define model
    model = Sequential()
    e = Embedding(vocab_size + 1, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH,
                  trainable=TRAINABLE_EMBEDDINGS, mask_zero=True)  # weights=[embedding_matrix]
    model.add(e)
    model.add(SpatialDropout1D(DROPOUT))
    model.add(LSTM(64, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    model.add(Dense(7))
    model.add(Activation("softmax"))

    # compile the model
    logger.info("Compiling model")
    model.compile(optimizer=Adam(LEARNING_RATE), loss=LOSS, metrics=['acc'])
    # summarize the model
    logger.info("Model Summary: %s", str(model.summary()))

    tensorboard_callback = TensorBoard(
        log_dir=TensorBoardLog, write_graph=True, write_images=True)

    # fit the model
    logger.info("Fitting model")
    model.fit(sentences_Train, labels_Train, batch_size=BATCH_SIZE,
              epochs=EPOCHS, callbacks=[tensorboard_callback])
    # evaluate the model
    logger.info("Evaluating model")
    loss, accuracy = model.evaluate(sentences_Test, labels_Test, verbose=0)
    logger.info('Accuracy: %f' % (accuracy * 100))
    logger.info("----End of Programm----")


def load_Data(dataset):
    utterances = []
    one_hot_intents = []

    if dataset == "dev":
        logger.info("Loading 'dev' Data")
        with open("TrainingData/Intents/dev.tsv", 'r') as devFile:
            for line in devFile:
                line = line.split("\t")

                utterance = line[1]  # [0] = intent, [1] = utterance

                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(
                    utterance, filters='?!,.:;\n', lower=False, split=' ')  # tokenizer, returns each utterance as list
                utterances.append(utterancePreProc)

                if int(line[0]) == 1:
                    encoded = np.array([1, 0, 0, 0, 0, 0, 0])
                elif int(line[0]) == 2:
                    encoded = np.array([0, 1, 0, 0, 0, 0, 0])
                elif int(line[0]) == 3:
                    encoded = np.array([0, 0, 1, 0, 0, 0, 0])
                elif int(line[0]) == 4:
                    encoded = np.array([0, 0, 0, 1, 0, 0, 0])
                elif int(line[0]) == 5:
                    encoded = np.array([0, 0, 0, 0, 1, 0, 0])
                elif int(line[0]) == 6:
                    encoded = np.array([0, 0, 0, 0, 0, 1, 0])
                elif int(line[0]) == 7:
                    encoded = np.array([0, 0, 0, 0, 0, 0, 1])

                one_hot_intents.append(encoded)
                npArray = np.array(one_hot_intents)
    elif dataset == "test":
        logger.info("Loading 'test' Data")
        with open("TrainingData/Intents/test.tsv", 'r') as devFile:
            for line in devFile:
                line = line.split("\t")

                utterance = line[1]

                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(
                    utterance, filters='?!,.:;\n', lower=False, split=' ')
                utterances.append(utterancePreProc)

                if int(line[0]) == 1:
                    encoded = np.array([1, 0, 0, 0, 0, 0, 0])
                elif int(line[0]) == 2:
                    encoded = np.array([0, 1, 0, 0, 0, 0, 0])
                elif int(line[0]) == 3:
                    encoded = np.array([0, 0, 1, 0, 0, 0, 0])
                elif int(line[0]) == 4:
                    encoded = np.array([0, 0, 0, 1, 0, 0, 0])
                elif int(line[0]) == 5:
                    encoded = np.array([0, 0, 0, 0, 1, 0, 0])
                elif int(line[0]) == 6:
                    encoded = np.array([0, 0, 0, 0, 0, 1, 0])
                elif int(line[0]) == 7:
                    encoded = np.array([0, 0, 0, 0, 0, 0, 1])

                one_hot_intents.append(encoded)
                npArray = np.array(one_hot_intents)
    else:
        logger.info("Loading 'train' Data")
        with open("TrainingData/Intents/train.tsv", 'r') as trainFile:
            for line in trainFile:
                line = line.split("\t")

                utterance = line[1]

                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(
                    utterance, filters='?!,.:;\n', lower=False, split=' ')
                utterances.append(utterancePreProc)

                if int(line[0]) == 1:
                    encoded = np.array([1, 0, 0, 0, 0, 0, 0])
                elif int(line[0]) == 2:
                    encoded = np.array([0, 1, 0, 0, 0, 0, 0])
                elif int(line[0]) == 3:
                    encoded = np.array([0, 0, 1, 0, 0, 0, 0])
                elif int(line[0]) == 4:
                    encoded = np.array([0, 0, 0, 1, 0, 0, 0])
                elif int(line[0]) == 5:
                    encoded = np.array([0, 0, 0, 0, 1, 0, 0])
                elif int(line[0]) == 6:
                    encoded = np.array([0, 0, 0, 0, 0, 1, 0])
                elif int(line[0]) == 7:
                    encoded = np.array([0, 0, 0, 0, 0, 0, 1])

                one_hot_intents.append(encoded)
                npArray = np.array(one_hot_intents)

    return utterances, npArray


def process_Data():
    tokenizer = Tokenizer()

    allSentences = []
    allSentencesGold = []

    sentencesTrain = []
    goldTrain = []
    sentencesDev = []
    goldDev = []
    sentencesTest = []
    goldTest = []

    if DATA_SET_TYPE == "dev":
        sentencesDev, goldDev = load_Data("dev")
        allSentences.extend(sentencesDev)
    else:
        sentencesDev, goldDev = load_Data("dev")
        sentencesTest, goldTest = load_Data("test")
        sentencesTrain, goldTrain = load_Data("train")

        allSentences.extend(sentencesDev)
        allSentences.extend(sentencesTest)
        allSentences.extend(sentencesTrain)

        allSentencesGold.extend(goldDev)
        allSentencesGold.extend(goldTest)
        allSentencesGold.extend(goldTrain)

    logger.warning("utterances: len train: %s, len test: %s, len dev: %s", len(
        sentencesTrain), len(sentencesTest), len(sentencesDev))
    logger.warning("gold: len train: %s, len test: %s, len dev: %s",
                   len(goldTrain), len(goldTest), len(goldDev))
    logger.warning("len all utterances: %s", len(allSentences))
    logger.warning("len all labels: %s", len(allSentences))

    tokenizer.fit_on_texts(allSentences)
    # + 3  # define the amount of tokens, +1 b/c of zero vector, ?+1 unknown?
    vocab_size = len(tokenizer.word_index)

    if DATA_SET_TYPE == "dev":
        logger.info("Splitting 'dev' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(
            sentencesDev, goldDev, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizer.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizer.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(
            X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(
            X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')

        return X_Train_Padded, X_Test_Padded, labels_Train, labels_Test, vocab_size, tokenizer

    else:
        logger.info("Splitting 'train' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(
            allSentences, allSentencesGold, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizer.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizer.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(
            X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(
            X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')

        return X_Train_Padded, X_Test_Padded, np.array(labels_Train), np.array(labels_Test), vocab_size, tokenizer


if __name__ == "__main__":
    main()
