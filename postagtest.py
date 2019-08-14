# https://nlpforhackers.io/lstm-pos-tagger-keras/

import numpy as np

from sklearn.model_selection import train_test_split
from keras import preprocessing
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam


def main ():
    print("Loading Stuff...")
    sentences = loadUtterances()
    sentence_tags = loadTags()
    
    print(sentence_tags)

    ##  Splits the sentences and tags into train- an test-sets
    (train_sentences, 
    test_sentences, 
    train_tags, 
    test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)


    ##  Create an index dictionary for sentences and tags
    words, tags = set([]), set([])
    
    for s in train_sentences:
        for w in s:
            words.add(w.lower())
    
    for ts in train_tags:
        for t in ts:
            tags.add(t)
    
    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
    
    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # The special value used to padding

    ##  Convert sentences and tags to numbers
    train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
    
    for s in train_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
    
        train_sentences_X.append(s_int)
    
    for s in test_sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
    
        test_sentences_X.append(s_int)
    
    for s in train_tags:
        train_tags_y.append([tag2index[t] for t in s])
    
    for s in test_tags:
        test_tags_y.append([tag2index[t] for t in s])
    
    print(train_sentences_X[0])
    print(test_sentences_X[0])
    print(train_tags_y[0])
    print(test_tags_y[0])


    ## Find max sentence length
    MAX_LENGTH = len(max(train_sentences_X, key=len))
    print(MAX_LENGTH)

    ## Pad the sentences and tags
    train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
    test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
    train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
    test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
    
    print(train_sentences_X[0])
    print(test_sentences_X[0])
    print(train_tags_y[0])
    print(test_tags_y[0])

    ## Define the model
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH, )))
    model.add(Embedding(len(word2index), 128, mask_zero=True))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(0.001),
                metrics=['accuracy'])
    
    model.summary()

    ## One-Hot-Enc for tags
    def to_categorical(sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
    print(cat_train_tags_y[0])

    ## Train
    model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=40, validation_split=0.2)

    ## Validate
    scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
    print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # acc: 99.09751977804825
 

def loadTags():
    with open("TrainingData/Slots/dev_slot_label.tsv", 'r') as devLabelFile:
        tagList = []
        for line in devLabelFile:
            line = line.replace("_", "")
            line = line.replace("\n", "")
            line = line.split(" ")
            tagList.append(line)
    return tagList

def loadUtterances():
    with open("TrainingData/Slots/dev_slot_Utt.tsv", 'r') as devUttFile:
        sentences = []
        for utterance in devUttFile:
            utterancePreProc = preprocessing.text.text_to_word_sequence(utterance, filters='?!,.:;\n', lower=False, split=' ')
            sentences.append(utterancePreProc)
    return sentences

if __name__ == "__main__":
    main()
    