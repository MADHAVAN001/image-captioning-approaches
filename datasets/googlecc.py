import numpy as np
from keras.utils import Sequence, to_categorical
from keras.preprocessing.sequence import pad_sequences
import os
import cv2
from cnn.Encoder import encoder_model
import itertools


tokenized_descriptions = 'tokenized_descriptions.txt'
word_dictionary = 'word_dictionary.txt'
vector_encoding = 'word_to_vector_encoding.txt'
index_replaced_file = ''


def prepare_dataset(cfg):
    if os.path.exists(os.path.join(cfg["workspace"]["directory"], index_replaced_file)):
        print("Word map already exists in workspace. Will be reused.")
        return

    indexes = dict()
    with open(cfg["data"]["googlecc"]["index_file"], 'r') as file:
        for line in file:
            values = line.strip().split(",")
            indexes[values[0]] = values[1]

    index_token_file = open(os.path.join(cfg["workspace"]["directory"], index_replaced_file), 'a')
    with open(cfg["data"]["googlecc"]["descriptions"]) as description_file:
        for line in description_file:
            values = line.strip().split(" ")

            if values[-1] in indexes.keys():
                values.insert(0, indexes[values[-1]])
                index_token_file.write(" ".join(values[:-1]))


def create_word_map(cfg):
    if os.path.exists(os.path.join(cfg["workspace"]["directory"], word_dictionary)):
        print("Word map already exists in workspace. Will be reused.")
        return

    print("Word map not found. Generating....")

    words_list = []
    words_to_id = {}

    with open(os.path.join(cfg["workspace"]["directory"], tokenized_descriptions), 'r') as file:
        for line in file:
            tokens = line.strip().split(",")
            words_list.extend(tokens[1:])

    # remove duplicate words
    words_list = list(set(words_list))

    # sorting the words
    words_list = sorted(words_list)
    for i in range(len(words_list)):
        words_to_id[words_list[i]] = i

    with open(os.path.join(cfg["workspace"]["directory"], word_dictionary), 'w') as f:
        [f.write('{0},{1}'.format(key, value) + "\n") for key, value in words_to_id.items()]


def tokenize_descriptions(cfg):
    if os.path.exists(os.path.join(cfg["workspace"]["directory"], tokenized_descriptions)):
        print("Tokenized descriptions found. Will not be generated.")
        return

    print("Generating tokenized descriptions")
    f = open(os.path.join(cfg["workspace"]["directory"], tokenized_descriptions), 'a')
    with open(cfg["data"]["googlecc"]["train_images_file"], 'r') as file:
        for line in file:
            if line.strip():
                sequence, link = line.split()[:-1]
                sequence.insert(0, '<START>')
                sequence.append('<END>')
                f.write(",".join(sequence)+"\n")
    f.close()
    print("Finished generating tokenized descriptions")
