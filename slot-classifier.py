# https://nlpforhackers.io/lstm-pos-tagger-keras/

import numpy as np
import logzero
from datetime import datetime
import os

from sklearn.model_selection import train_test_split
from keras import preprocessing
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import backend as K

logzero.logfile("Log/slots-logs.log", maxBytes=1e6, backupCount=5)
logzero.loglevel(10)
logger = logzero.logger

TensorBoardLog = "Log/tensorboard/slots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.mkdir(TensorBoardLog)

DATA_SET_TYPE = "train"

LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCHS = 15 
EMBEDDING_SIZE = 800
TRAINABLE_EMBEDDINGS = True

logger.info("----Start of Program----")
logger.debug("Learning Rate: %s ", LEARNING_RATE)
logger.debug("Batch Size: %s ", BATCH_SIZE)
logger.debug("Epochs: %s ", EPOCHS)
logger.debug("Data Set Type: %s", DATA_SET_TYPE)
logger.debug("Embedding Size: %s", EMBEDDING_SIZE)
logger.debug("Trainable Embedding: %s", TRAINABLE_EMBEDDINGS)

def main ():
    utterances = loadUtterances()
    utterance_slots = loadSlots() 
    
    ##  Splits the utterances and slots into train- an test-sets
    (train_utterances, 
    test_utterances, 
    train_slots, 
    test_slots) = train_test_split(utterances, utterance_slots, test_size=0.33, random_state=42)

    ##  Create an index dictionary for utterances and slots
    words_set, slots_set = set([]), set([]) 
    
    for utterance in train_utterances:
        for word in utterance:
            words_set.add(word)
    
    for slots in train_slots:
        for slot in slots:
            slots_set.add(slot)
    
    word2index = {word: i + 2 for i, word in enumerate(list(words_set))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
    
    slot2index = {slot: i + 1 for i, slot in enumerate(list(slots_set))}
    slot2index['-PAD-'] = 0  # The special value used to padding

    ##  Convert utterances and slots to numbers
    train_utterances_X, test_utterances_X, train_slots_Y, test_slots_Y = [], [], [], []
    
    for utterance in train_utterances:
        utterance_int = [] # utterance with indices
        for word in utterance:
            try:
                utterance_int.append(word2index[word])
            except KeyError:
                utterance_int.append(word2index['-OOV-'])
    
        train_utterances_X.append(utterance_int)
    
    for utterance in test_utterances:
        utterance_int = []
        for word in utterance:
            try:
                utterance_int.append(word2index[word])
            except KeyError:
                utterance_int.append(word2index['-OOV-'])
    
        test_utterances_X.append(utterance_int)
    
    for utterance in train_slots:
        train_slots_Y.append([slot2index[slot] for slot in utterance])
    
    for utterance in test_slots:
        test_slots_Y.append([slot2index[slot] for slot in utterance])

    # logger.debug("Train Utterance: %s", train_utterances_X[0])
    # logger.debug("Test Utterance: %s", test_utterances_X[0])
    # logger.debug("Train Slots: %s", train_slots_Y[0])
    # logger.debug("Test Slots: %s", test_slots_Y[0])

    ## Find max sentence length for padding
    MAX_LENGTH = len(max(train_utterances_X, key=len))
    logger.debug("Maximum utterance lenght: %s", MAX_LENGTH)

    ## Pad the utterances and slots (keras)
    train_utterances_X = pad_sequences(train_utterances_X, maxlen=MAX_LENGTH, padding='post')
    test_utterances_X = pad_sequences(test_utterances_X, maxlen=MAX_LENGTH, padding='post')
    train_slots_Y = pad_sequences(train_slots_Y, maxlen=MAX_LENGTH, padding='post')
    test_slots_Y = pad_sequences(test_slots_Y, maxlen=MAX_LENGTH, padding='post')
    
    # logger.debug("Train Utterance: %s", train_utterances_X[0])
    # logger.debug("Test Utterance: %s", test_utterances_X[0])
    # logger.debug("Train Slots: %s", train_slots_Y[0])
    # logger.debug("Test Slots: %s", test_slots_Y[0])

    ## Define the model
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH, )))
    model.add(Embedding(len(word2index), EMBEDDING_SIZE, mask_zero=True, trainable=TRAINABLE_EMBEDDINGS))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(slot2index))))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(LEARNING_RATE),
                metrics=['accuracy'])
    
    model.summary()

    tensorboard_callback = TensorBoard(log_dir=TensorBoardLog, write_graph=True, write_images=True)

    cat_train_slots_y = to_categorical(train_slots_Y, len(slot2index))
    # logger.debug("Slots to categorical: %s", cat_train_slots_y[0])

    cat_test_slots_y = to_categorical(test_slots_Y, len(slot2index))

    ## Train
    model.fit(train_utterances_X, cat_train_slots_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tensorboard_callback], validation_split=0.2)

    ## Validate
    scores = model.evaluate(test_utterances_X, cat_test_slots_y)
    print(f"{model.metrics_names[1]}: {scores[1] * 100}")
    logger.info('Accuracy: %f' % (scores[1]*100))
    logger.info("----End of Programm----")
 
## One-Hot-Enc for tags
def to_categorical(sequences, categories):
    cat_sequences = []
    for utterance in sequences:
        cats = []
        for item in utterance:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def loadSlots():
    if DATA_SET_TYPE == "dev":
        with open("TrainingData/Slots/dev_slot_label.tsv", 'r') as devLabelFile:
            slotList = []
            for line in devLabelFile:
                line = line.replace("_", "")
                line = line.replace("\n", "")
                line = line.split("\t")
                slotList.append(line)
    else:
        Data_Path_List = ["TrainingData/Slots/dev_slot_label.tsv", "TrainingData/Slots/train_slot_label.tsv", "TrainingData/Slots/test_slot_label.tsv"]
        slotList = []
        for path in Data_Path_List:
            with open(path, 'r') as devLabelFile:
                    for line in devLabelFile:
                        line = line.replace("_", "")
                        line = line.replace("\n", "")
                        line = line.split("\t")
                        slotList.append(line)
    
    return slotList

def loadUtterances():
    if DATA_SET_TYPE == "dev":
        with open("TrainingData/Slots/dev_slot_Utt.tsv", 'r') as devUttFile:
            utterances = []
            for utterance in devUttFile:
                utterancePreProc = preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split='\t')
                utterances.append(utterancePreProc)
    else:
        Data_Path_List = ["TrainingData/Slots/dev_slot_Utt.tsv", "TrainingData/Slots/train_slot_Utt.tsv", "TrainingData/Slots/test_slot_Utt.tsv"]
        utterances = []
        for path in Data_Path_List:
            with open(path, 'r') as devUttFile:
                for utterance in devUttFile:
                    utterancePreProc = preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split='\t')
                    utterances.append(utterancePreProc)
    
    return utterances

if __name__ == "__main__":
    main()
    