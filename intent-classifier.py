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
# import msgpack
# import msgpack_numpy as msgp
# msgp.patch()

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from bert_embedding import BertEmbedding

from sklearn.model_selection import train_test_split

DATA_SET_TYPE = "dev"

MAX_SENTENCE_LENGTH = 30
EMBEDDING_SIZE = 768

EPOCHS = 50
BATCH_SIZE = 30
ACTIVATION = ""
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
DROPOUT = 0.4

logzero.logfile("Log/logs.log", maxBytes=1e6, backupCount=5)
logzero.loglevel(10)
logger = logzero.logger

def main():
    logger.info("----Start of Program----")

    bert_embedding = BertEmbedding()

    sentences_Train, sentences_Test, labels_Train, labels_Test, vocab_size, tokenizer = process_Data()

    logger.warning("vocab_size: %s", vocab_size)

    # X_train, Y_train = load_Dev()
   
    # if os.path.isfile("Cache/embeddingMatrix.pickle"):
    if os.path.isfile("Cache/embeddingMatrix.pickle"):
        logger.info("Loading existing embedding matrix")
        with open("Cache/embeddingMatrix.pickle", "rb") as file:
            embedding_matrix = pickle.load(file)
            # embedding_matrix = msgpack.unpackb(file, default=msgp.encode)
        logger.info("Finished Loading Embedding Matrix")

    else: 
        logger.info("Creating Embedding Matrix")
        # create a weight matrix for words in training utterancePreProc
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_SIZE))

        for word, i in tokenizer.word_index.items():
            # Creating a list on which to put the word since the python implementation of BERT want's a list not a single word
            tmpList = []
            tmpList.append(word)
            #logger.info("Search Embedding: %s", word)
            embedding_vector = bert_embedding(tmpList)
            #logger.info("Found Embedding")
            # print(embedding_vector[0][1][0])
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector[0][1][0]

        embedding_matrix = np.insert(embedding_matrix, 0, np.zeros(EMBEDDING_SIZE), axis=0)

        print(embedding_matrix)

        with open("Cache/embeddingMatrix.pickle", "wb") as file:
            pickle.dump(embedding_matrix, file)
            # packed = msgpack.packb(embedding_matrix, default=msgp.encode)
            # file.write(packed)

        logger.info("Finished Embedding Matrix Creation, dumped it.")
    # define model
    model = Sequential()
    e = Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH, weights=[embedding_matrix], trainable=False, mask_zero=True)
    model.add(e)
    model.add(SpatialDropout1D(DROPOUT))
    model.add(LSTM(64, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    model.add(Dense(7))
    model.add(Activation("softmax"))
    #model.add(Dense(7, activation='softmax'))

    # compile the model
    logger.info("Compiling model")
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['acc'])
    # summarize the model
    logger.info("Model Summary: %s", str(model.summary()))
    # fit the model
    logger.info("Fitting model")
    model.fit(sentences_Train, labels_Train, epochs=EPOCHS, verbose=0)
    # evaluate the model
    logger.info("Evaluating model")
    loss, accuracy = model.evaluate(sentences_Test, labels_Test, verbose=0)
    logger.info('Accuracy: %f' % (accuracy*100))
    logger.info("----End of Programm----")


def load_Data(dataset):
    sentences = []
    one_hot_intents = []

    if dataset == "dev":
        logger.info("Loading 'dev' Data")
        with open("TrainingData/Intents/dev.tsv", 'r') as devFile:
            for line in devFile:
                line = line.split("\t")
                
                utterance = line[1]
                
                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split=' ')
                sentences.append(utterancePreProc)

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
                
                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split=' ')
                sentences.append(utterancePreProc)

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
                
                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split=' ')
                sentences.append(utterancePreProc)

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
                
    return sentences, npArray


def process_Data():
    tokenizer = Tokenizer()

    allSentences = []

    sentencesTrain, goldTrain = load_Data("train")
    sentencesTest, goldTest = load_Data("test")
    sentencesDev, goldDev = load_Data("dev")

    logger.warning("sentences: len train: %s, len test: %s, len dev: %s", len(sentencesTrain), len(sentencesTest), len(sentencesDev))
    logger.warning("gold: len train: %s, len test: %s, len dev: %s", len(goldTrain), len(goldTest), len(goldDev))

    allSentences.extend(sentencesDev)
    allSentences.extend(sentencesTest)
    allSentences.extend(sentencesTrain)

    logger.warning("len all sentences: %s", len(sentencesTrain))

    tokenizer.fit_on_texts(allSentences)
    vocab_size = len(tokenizer.word_index) + 2  # define the amount of tokens, +1 b/c of zero vector, ?+1 unknown?

    if DATA_SET_TYPE == "dev":
        logger.info("Splitting 'dev' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(sentencesDev, goldDev, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizer.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizer.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')

        return X_Train_Padded, X_Test_Padded, labels_Train, labels_Test, vocab_size, tokenizer

    elif DATA_SET_TYPE == "test":
        logger.info("Splitting 'test' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(sentencesTest, goldTest, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizer.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizer.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')

        return X_Train_Padded, X_Test_Padded, labels_Train, labels_Test, vocab_size, tokenizer

    else:
        logger.info("Splitting 'train' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(sentencesTrain, goldTrain, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizer.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizer.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')

        return X_Train_Padded, X_Test_Padded, labels_Train, labels_Test, vocab_size, tokenizer


if __name__ == "__main__":
    main()
