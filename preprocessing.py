#!/usr/bin/env python3
"""
intent-preprocessing.py

Preprocesses the data

*full.json => train.tsv
validate_*.json => test.tsv
*.json => dev.tsv

1= BookRestaurant
2 = SearchScreeningEvent
3 = RateBook
4 = PlayMusic
5 = AddToPlaylist
6 = GetWeather
7 = SearchCreativeWork

@Author: Fabian Fey, Gianna Weber
"""

import os
import json
import codecs

path_train=[]
path_dev=[]
path_test=[]
intent_dict = {"BookRestaurant":1,
"SearchScreeningEvent":2,
"RateBook":3,
"PlayMusic":4,
"AddToPlaylist":5,
"GetWeather":6,
"SearchCreativeWork":7
}


def removeChar(str, n):
    first_part = str[:n] 
    last_part = str[n+1:]
    return first_part + last_part


for subdir, dirs, files in os.walk('Data/'):
    for file in files:
        if "train" in file and "full" not in file:  # dev
            # print(os.path.join(subdir, file))
            path_dev.append(os.path.join(subdir, file))
        elif "full" in file:  # test
            path_test.append(os.path.join(subdir, file))
        elif "validate" in file:  # train
            path_train.append(os.path.join(subdir, file))
# print(paths)


def intents(paths, filename):
    with open(filename, 'w') as OutputFile:
        for path in paths:
            with open(path, 'r', encoding='ISO-8859-1') as jsonFile:
                print(path)
                file = json.load(jsonFile)
                for intent in file:  # {"AddToPlaylist":[...]}
                    #print(intent)
                    for data in file[intent]:  # {"AddToPlaylist":[{"data": [...]}, {...}, ...]}
                        #print(data)
                        for key in data:
                            # print(key)
                            text = []
                            entity = []
                            index = 0
                            for entry in data[key]:
                                #print(entry["text"])
                                tmpText = entry["text"]
                                tmpText = tmpText.replace("\n", "")
                                tmpText = tmpText.replace("  ", " ")
                                text.append(tmpText)
                            if len("".join(text)) != 0:
                                print(str(intent_dict[intent])+"\t"+"".join(text))
                                OutputFile.write(str(intent_dict[intent])+"\t"+"".join(text)+"\n")


def slots(paths, slotLabel, slotUtterance):
    with open(slotLabel, 'w') as LabelOutputFile:
        with open(slotUtterance, 'w') as UtteranceOutputFile:
            for path in paths:
                with open(path, 'r', encoding='ISO-8859-1') as jsonFile:
                    file = json.load(jsonFile)
                    for intent in file:  # {"AddToPlaylist":[...]}
                        # print(intent)
                        for data in file[intent]:  # {"AddToPlaylist":[{"data": [...]}, {...}, ...]}
                            # print(data)
                            for key in data:
                                # print(key)
                                text = []
                                entity = []
                                for entry in data[key]:
                                    entryText = []
                                    if "entity" in entry:
                                        utterance = entry["text"]

                                        utterance = utterance.replace("\n", "")
                                        utterance = utterance.replace("  ", " ")
                                        textSplit = utterance.split(" ")
                                        
                                        for i in range(0, len(textSplit)):
                                            entity.append(entry["entity"])
                                        text.extend(textSplit)
                                        # print("Text: %s", text)
                                    else:
                                        utterance = entry["text"]
                                        utterance = utterance.replace("\n", "")
                                        utterance = utterance.replace("  ", " ")
                                        
                                        if utterance == " ":
                                            continue
                                        elif utterance[:1] == " ":
                                            if utterance[-1:] == " ":
                                                tmp = removeChar(utterance, 0)
                                                tmp2 = removeChar(tmp, len(tmp)-1)
                                                entryText.extend(tmp2.split(" "))
                                            else:
                                                tmp = removeChar(utterance, 0)
                                                entryText.extend(tmp.split(" "))
                                        elif utterance[-1:] == " ":
                                            tmp = removeChar(utterance, len(utterance)-1)
                                            entryText.extend(tmp.split(" "))
                                        else:
                                            entryText.extend(utterance.split(" "))

                                        for i in range(0, len(entryText)):
                                            entity.append("NULL")
                                    
                                    text.extend(entryText)

                                #print("".join(text))
                                utterance = " ".join(text)
                                labels = " ".join(entity)
                                
                                utterance = utterance.replace("\n", "")
                                utterance = utterance.replace("  ", " ")
                                labels = labels.replace("\n", "")
                                
                                LabelOutputFile.write(labels + "\n")
                                UtteranceOutputFile.write(utterance + "\n") 

def slotCount(path_dev, path_test, path_train):
    paths = []
    paths.extend(path_dev)
    paths.extend(path_test)
    paths.extend(path_train)
    SlotDict = {}
    for path in paths:
        with open(path, 'r', encoding='ISO-8859-1') as jsonFile:
            file = json.load(jsonFile)
            for intent in file:  # {"AddToPlaylist":[...]}
                # print(intent)
                for data in file[intent]:  # {"AddToPlaylist":[{"data": [...]}, {...}, ...]}
                    # print(data)
                    for key in data:
                        # print(key)
                        text = []
                        entity = []
                        for entry in data[key]:
                            entryText = []
                            if "entity" in entry:
                                slot = entry["entity"]
                                slot = slot.replace("_", "")
                                if slot in SlotDict:
                                    SlotDict[slot] +=1
                                else:
                                    SlotDict[slot] = 1
                            else:
                                if "NULL" in SlotDict:
                                    SlotDict["NULL"] +=1
                                else:
                                    SlotDict["NULL"] = 1

    with open("SlotList.tsv", "w") as slotFile:
        for key in SlotDict:
            slotFile.write(str(key+"\t"))
            print(key, SlotDict[key]) # Slot Verteilung
        print(len(SlotDict)) # 39 entity, i.e. slot, types

       
slotCount(path_dev, path_test, path_train)
# slots(path_dev, "dev_slot_label.tsv", "dev_slot_Utt.tsv")
# slots(path_test, "test_slot_label.tsv", "test_slot_Utt.tsv")
# slots(path_train, "train_slot_label.tsv", "train_slot_Utt.tsv")
# intents(path_dev, "dev.tsv")
# intents(path_test, "test.tsv")
# intents(path_train, "train.tsv")