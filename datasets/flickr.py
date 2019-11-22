import itertools
import os

import cv2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence, to_categorical

from cnn.Encoder import encoder_model
from datasets.common import tokenize_descriptions, read_word_dictionary, vector_encode_descriptions, create_word_map, \
    read_encoded_descriptions, read_id_to_word_dictionary, get_dataset_metadata_cfg

tokenized_descriptions = 'tokenized_descriptions.txt'
word_dictionary = 'word_dictionary.txt'
vector_encoding = 'word_to_vector_encoding.txt'


def image_generator(cfg, data_list):
    dataset_metadata = get_dataset_metadata_cfg()
    with open(data_list, 'r') as file:
        for line in file:
            image = cv2.imread(os.path.join(dataset_metadata["data"]["flickr"]["dataset_path"], line.strip()))
            image = np.expand_dims(np.asarray(cv2.resize(image, (299, 299))) / 255.0, axis=0)
            yield image


def retrieve_data_list_file(cfg, run_type):
    dataset_cfg = get_dataset_metadata_cfg()

    if run_type == "train":
        data_list = dataset_cfg["data"]["flickr"]["train_images_file"]
    elif run_type is "validation":
        data_list = dataset_cfg["data"]["flickr"]["validation_images_file"]
    else:
        data_list = dataset_cfg["data"]["flickr"]["test_images_file"]

    return data_list


def get_line_count(file):
    return sum(1 for line in open(file))


def encode_images(cfg, unique_id, model, run_type):
    if os.path.exists(os.path.join(cfg["workspace"]["directory"], unique_id + "_" + run_type + "_encoding.npy")):
        print("Image encodings found for {}:{}".format(unique_id, run_type))
        return

    data_list = retrieve_data_list_file(cfg, run_type)

    model = encoder_model(model)
    enc_train = model.predict_generator(image_generator(cfg, data_list), steps=get_line_count(data_list), verbose=1)
    print("Number of encodings: " + str(len(enc_train)))
    np.save(os.path.join(cfg["workspace"]["directory"], unique_id + "_" + run_type + "_encoding.npy"), enc_train)


class PreProcessing:

    def __init__(self, cfg):
        dataset_cfg = get_dataset_metadata_cfg()

        self.workspace_dir = cfg["workspace"]["directory"]
        self.tokenized_descriptions_file_path = os.path.join(self.workspace_dir, tokenized_descriptions)
        self.word_dictionary_file_path = os.path.join(self.workspace_dir, word_dictionary)
        self.vector_encoding_file_path = os.path.join(self.workspace_dir, vector_encoding)
        self.cfg = cfg

        tokenize_descriptions(dataset_cfg["data"]["flickr"]["descriptions"], self.tokenized_descriptions_file_path)
        create_word_map(self.tokenized_descriptions_file_path, self.word_dictionary_file_path)

        word_map = read_word_dictionary(self.word_dictionary_file_path)
        vector_encode_descriptions(self.tokenized_descriptions_file_path, self.vector_encoding_file_path, word_map)

    def get_word_dictionary(self):
        return read_word_dictionary(self.word_dictionary_file_path)

    def get_id_dictionary(self):
        return read_id_to_word_dictionary(self.word_dictionary_file_path)

    def get_vector_encodings(self):
        return read_encoded_descriptions(self.vector_encoding_file_path)

    def get_keras_generators(self, unique_id):
        return FlickrDataGenerator(self.cfg, unique_id, "train", self.vector_encoding_file_path), \
               FlickrDataGenerator(self.cfg, unique_id, "validation", self.vector_encoding_file_path),\
               FlickrDataGenerator(self.cfg, unique_id, "test", self.vector_encoding_file_path)


class FlickrDataGenerator(Sequence):

    def __init__(self, cfg, unique_id, run_type, vector_encoding_file_path):

        if run_type == "train":
            self.batch_size = cfg["training"]["batch_size"]
        elif run_type == "validation":
            self.batch_size = cfg["validation"]["batch_size"]
        else:
            self.batch_size = 1

        self.token_count = get_line_count(os.path.join(cfg["workspace"]["directory"], word_dictionary))
        self.word_to_vec_map = read_encoded_descriptions(vector_encoding_file_path)
        self.data_list = retrieve_data_list_file(cfg, run_type)
        self.image_encoding = np.load(os.path.join(cfg["workspace"]["directory"], unique_id + "_" + run_type + "_encoding.npy"))
        self.previous_line_number = 0
        self.max_sentence_len = 40
        self.vocab_size = get_line_count(os.path.join(cfg["workspace"]["directory"], word_dictionary))

    def __len__(self):
        return int(np.ceil(get_line_count(self.data_list)/float(self.batch_size)))

    def __getitem__(self, idx):

        batch_images = list()
        batch_word_sequences = list()
        batch_output = list()

        with open(self.data_list) as f:
            result = itertools.islice(f, idx*self.batch_size, (idx+1)*self.batch_size)

            index = idx*self.batch_size
            for line in result:

                for i in range(5):
                    sequence = self.word_to_vec_map[line.strip()+"#"+str(i)]

                    for j in range(1, len(sequence)):
                        input_word_sequence, output_word_sequence = sequence[:j], sequence[j]

                        input_word_sequence = pad_sequences(
                            [list(map(int, input_word_sequence))],
                            maxlen=self.max_sentence_len,
                            padding='post'
                        )[0]
                        output_word_sequence = to_categorical(
                            [int(output_word_sequence)],
                            num_classes=self.vocab_size
                        )[0]

                        batch_images.append(self.image_encoding[index])
                        batch_word_sequences.append(input_word_sequence)
                        batch_output.append(output_word_sequence)
                index += 1

        return [np.array(batch_images), np.array(batch_word_sequences)], np.array(batch_output)
