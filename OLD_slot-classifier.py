#!/usr/bin/env python3
"""
intent-classifier.py

Embedding Layer:
https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
https://github.com/CyberZHG/keras-bert

@Author: Fabian Fey, Gianna Weber
"""

from collections import Counter


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
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, LSTM, SpatialDropout1D, SimpleRNN, InputLayer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from bert_embedding import BertEmbedding

from sklearn.model_selection import train_test_split

DATA_SET_TYPE = "dev"

MAX_SENTENCE_LENGTH = 30
EMBEDDING_SIZE = 768

EPOCHS = 50
# BATCH_SIZE = 30
ACTIVATION = ""
OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
DROPOUT = 0.4

logzero.logfile("Log/slot-logs.log", maxBytes=1e6, backupCount=5)
logzero.loglevel(10)
logger = logzero.logger

def main():
    logger.info("----Start of Program----")

    bert_embedding = BertEmbedding()

    sentences_Train, sentences_Test, labels_Train, labels_Test, vocab_size, tokenizerSentences = process_Data()

    LabelsTrainPadded = []
    LabelsTestPadded = []
    
    for utt in labels_Train:
        tmp = []
        for label in utt:
            # print(keras.utils.to_categorical(label, 41))
            tmp.extend(keras.utils.to_categorical(label, 41))  # 40 slots + NULL
        LabelsTrainPadded.append(tmp)
    for utt in labels_Test:
        tmp = []
        for label in utt:
            tmp.extend(keras.utils.to_categorical(label, 41))  # 40 slots + NULL
        LabelsTestPadded.append(tmp)
    
    logger.warning("vocab_size: %s", vocab_size)
   
    # if os.path.isfile("Cache/slotEmbeddingMatrix.pickle"):
    if os.path.isfile("Cache/slotEmbeddingMatrix.pickle"):
        logger.info("Loading existing embedding matrix")
        with open("Cache/slotEmbeddingMatrix.pickle", "rb") as file:
            embedding_matrix = pickle.load(file)
            # embedding_matrix = msgpack.unpackb(file, default=msgp.encode)
        logger.info("Finished Loading Embedding Matrix")

    else: 
        logger.info("Creating Embedding Matrix")
        # create a weight matrix for words in training utterancePreProc
        embedding_matrix = np.zeros((vocab_size, EMBEDDING_SIZE))

        for word, i in tokenizerSentences.word_index.items():
            # Creating a list on which to put the word since the python implementation of BERT want's a list not a single word
            tmpList = []
            tmpList.append(word)
            #logger.info("Search Embedding: %s", word)
            embedding_vector = bert_embedding(tmpList)
            #logger.info("Found Embedding")
            # print(embedding_vector[0][1][0])
            if embedding_vector is not None:
                try:
                    embedding_matrix[i] = embedding_vector[0][1][0]
                except:
                    logger.exception("i: %s", i)

        embedding_matrix = np.insert(embedding_matrix, 0, np.zeros(EMBEDDING_SIZE), axis=0)

        print(embedding_matrix)

        with open("Cache/slotEmbeddingMatrix.pickle", "wb") as file:
            pickle.dump(embedding_matrix, file)
            # packed = msgpack.packb(embedding_matrix, default=msgp.encode)
            # file.write(packed)

        logger.info("Finished Embedding Matrix Creation, dumped it.")
    # define model
    model = Sequential()
    #model.add(InputLayer(input_shape=(30,)))
    e = Embedding(vocab_size + 1, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH, weights=[embedding_matrix], trainable=False, mask_zero=True)
    model.add(e)
    model.add(SpatialDropout1D(DROPOUT))
    model.add(LSTM(64, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    model.add(Dense(30))
    model.add(Activation("softmax"))

    # compile the model
    logger.info("Compiling model")
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['acc'])
    # summarize the model
    logger.info("Model Summary: %s", str(model.summary()))
    # fit the model
    logger.info("Fitting model")
    # print(type(LabelsTestPadded))
    # logger.warning("Sentences %s", LabelsTestPadded.shape)
    # logger.warning("Labels %s", LabelsTrainPadded.shape)
    # print(np.array(LabelsTrainPadded))
    # print(len(LabelsTrainPadded[0][0]))
    # print(labels_Train)
    # labels_Train.reshape(1407, 30)
    # print(labels_Train.shape)
    print(labels_Train)
    model.fit(sentences_Train, labels_Train, epochs=EPOCHS, verbose=0)
    # evaluate the model
    logger.info("Evaluating model")
    loss, accuracy = model.evaluate(sentences_Test, labels_Test, verbose=0)
    logger.info('Accuracy: %f' % (accuracy*100))
    logger.info("----End of Programm----")

def load_Data(dataset):
    tokenizerSlots = Tokenizer()

    sentences = []
    slotIndices = []
    slotList = LoadSlotList()

    uniqueTokens = []

    tokenizerSlots.fit_on_texts(slotList)

    if dataset == "dev":
        logger.info("Loading 'dev' Data")
        with open("TrainingData/Slots/dev_slot_label.tsv", 'r') as devLabelFile:
            for line in devLabelFile:
                # line = keras.preprocessing.text.text_to_word_sequence(line, filters='\n', lower=False, split='\t')
                line = line.replace("\n", "")
                line = line.replace("_", "")  # Remove underscores in slots b/c 'text_to_sequences' stupidly splits again at underscores... 'party_size_number' --> 'partysizenumber'
                line = line.split(" ")
                uniqueTokens.extend(line)
                sequence = tokenizerSlots.texts_to_sequences(line)
                
                # flattened = [val for sublist in sequence for val in sublist]  # flatten the strange nested list ourput from the 'texts_to_sequences()'
                slotIndices.append(sequence)
        
        with open("TrainingData/Slots/dev_slot_Utt.tsv", 'r') as devUttFile:
            for utterance in devUttFile:

                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split=' ')
                sentences.append(utterancePreProc)

    elif dataset == "test":
        logger.info("Loading 'test' Data")

        with open("TrainingData/Slots/test_slot_label.tsv", 'r') as devLabelFile:
            for line in devLabelFile:
                line = line.replace("\n", "")
                line = line.replace("_", "")  # Remove underscores in slots b/c 'text_to_sequences' stupidly splits again at underscores... 'party_size_number' --> 'partysizenumber'
                line = line.split(" ")
                uniqueTokens.extend(line)
                sequence = tokenizerSlots.texts_to_sequences(line)

                # flattened = [val for sublist in sequence for val in sublist]  # flatten the strange nested list ourput from the 'texts_to_sequences()'
                slotIndices.append(sequence)

        with open("TrainingData/Slots/test_slot_Utt.tsv", 'r') as devUttFile:
            for utterance in devUttFile:

                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split=' ')
                sentences.append(utterancePreProc)

    else:
        logger.info("Loading 'train' Data")
        with open("TrainingData/Slots/train_slot_label.tsv", 'r') as devLabelFile:
            for line in devLabelFile:
                # line = keras.preprocessing.text.text_to_word_sequence(line, filters='\n', lower=False, split='\t')
                line = line.replace("\n", "")
                line = line.replace("_", "")  # Remove underscores in slots b/c 'text_to_sequences' stupidly splits again at underscores... 'party_size_number' --> 'partysizenumber'
                line = line.split(" ")
                uniqueTokens.extend(line)
                sequence = tokenizerSlots.texts_to_sequences(line)

                # flattened = [val for sublist in sequence for val in sublist]  # flatten the strange nested list ourput from the 'texts_to_sequences()'
                slotIndices.append(sequence)

        with open("TrainingData/Slots/train_slot_Utt.tsv", 'r') as devUttFile:
            for utterance in devUttFile:

                utterancePreProc = keras.preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split=' ')
                sentences.append(utterancePreProc)

    logger.debug("All unique slots: %s", Counter(uniqueTokens).keys())  # equals to list(set(words))

    nullIndex = tokenizerSlots.texts_to_sequences(['NULL'])[0][0]
    return sentences, slotIndices, nullIndex

def LoadSlotList():

    with open("TrainingData/Slots/SlotList.tsv", "r") as LoadSlotFile:
        slotList = []
        for line in LoadSlotFile:
            lineSplit = line.split("\t")
            for item in lineSplit:
                if item != "":
                    slotList.append(item)

        # logger.debug("all slots: %s", slotList)
        logger.debug("Lenght SlotList: %s", len(slotList))
    
    return(slotList)

def process_Data():
    tokenizerSentences = Tokenizer()

    allSentences = []

    sentencesTrain, goldTrain, nullIndex = load_Data("train")
    sentencesTest, goldTest, nullIndex = load_Data("test")
    sentencesDev, goldDev, nullIndex = load_Data("dev")

    logger.warning("sentences: len train: %s, len test: %s, len dev: %s", len(sentencesTrain), len(sentencesTest), len(sentencesDev))
    logger.warning("gold: len train: %s, len test: %s, len dev: %s", len(goldTrain), len(goldTest), len(goldDev))
    
    allSentences.extend(sentencesDev)
    allSentences.extend(sentencesTest)
    allSentences.extend(sentencesTrain)

    logger.warning("len all sentences: %s", len(allSentences))
    # logger.warning("Sentences Train: %s", sentencesTrain)
    tokenizerSentences.fit_on_texts(allSentences)
    vocab_size = len(tokenizerSentences.word_index)  # + 2  # define the amount of tokens, +1 b/c of zero vector, ?+1 unknown?

    if DATA_SET_TYPE == "dev":
        logger.info("Splitting 'dev' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(sentencesDev, goldDev, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizerSentences.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizerSentences.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        
        Y_Train_Padded = pad_sequences(labels_Train, maxlen=MAX_SENTENCE_LENGTH, padding='post', value = int(nullIndex))
        Y_Test_Padded = pad_sequences(labels_Test, maxlen=MAX_SENTENCE_LENGTH, padding='post', value = int(nullIndex))

        return X_Train_Padded, X_Test_Padded, Y_Train_Padded, Y_Test_Padded, vocab_size, tokenizerSentences

    elif DATA_SET_TYPE == "test":
        logger.info("Splitting 'test' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(sentencesTest, goldTest, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizerSentences.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizerSentences.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')

        Y_Train_Padded = pad_sequences(labels_Train, maxlen=MAX_SENTENCE_LENGTH, padding='post', value = int(nullIndex))
        Y_Test_Padded = pad_sequences(labels_Test, maxlen=MAX_SENTENCE_LENGTH, padding='post', value = int(nullIndex))

        return X_Train_Padded, X_Test_Padded, Y_Train_Padded, Y_Test_Padded, vocab_size, tokenizerSentences

    else:
        logger.info("Splitting 'train' Data")
        sentences_Train, sentences_Test, labels_Train, labels_Test = train_test_split(sentencesTrain, goldTrain, test_size=0.33, random_state=42)

        # Create an int encoded version of each token
        X_Train_Encoded = tokenizerSentences.texts_to_sequences(sentences_Train)
        X_Test_Encoded = tokenizerSentences.texts_to_sequences(sentences_Test)

        # Pad each sentence to a length of MAX_SENTENCE_LENGTH
        X_Train_Padded = pad_sequences(X_Train_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')
        X_Test_Padded = pad_sequences(X_Test_Encoded, maxlen=MAX_SENTENCE_LENGTH, padding='post')

        Y_Train_Padded = pad_sequences(labels_Train, maxlen=MAX_SENTENCE_LENGTH, padding='post', value = int(nullIndex))
        Y_Test_Padded = pad_sequences(labels_Test, maxlen=MAX_SENTENCE_LENGTH, padding='post', value = int(nullIndex))

        return X_Train_Padded, X_Test_Padded, Y_Train_Padded, Y_Test_Padded, vocab_size, tokenizerSentences

if __name__ == "__main__":
    main()
