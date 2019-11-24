import numpy as np
from keras.utils import Sequence, to_categorical
from keras.preprocessing.sequence import pad_sequences
import os
import cv2
from cnn.Encoder import encoder_model
import itertools
from datasets.common import get_dataset_metadata_cfg, tokenize_descriptions, vector_encode_descriptions, \
    read_encoded_descriptions, read_id_to_word_dictionary, read_word_dictionary, create_word_map


tokenized_descriptions = 'tokenized_descriptions.txt'
word_dictionary = 'word_dictionary.txt'
vector_encoding = 'word_to_vector_encoding.txt'
index_replaced_file = 'google_cc_processed_descriptions.txt'


def retrieve_data_list_file(run_type):
    dataset_cfg = get_dataset_metadata_cfg()

    if run_type == "train":
        data_list = dataset_cfg["data"]["googlecc"]["train_images_file"]
    elif run_type is "validation":
        data_list = dataset_cfg["data"]["googlecc"]["validation_images_file"]
    else:
        data_list = dataset_cfg["data"]["googlecc"]["test_images_file"]

    return data_list


def get_line_count(file):
    return sum(1 for line in open(file))


class PreProcessing:

    def __init__(self, cfg, unique_id):
        self.dataset_cfg = get_dataset_metadata_cfg()

        self.workspace_dir = os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"])

        self.tokenized_descriptions_file_path = os.path.join(self.workspace_dir, tokenized_descriptions)
        self.word_dictionary_file_path = os.path.join(self.workspace_dir, word_dictionary)
        self.vector_encoding_file_path = os.path.join(self.workspace_dir, vector_encoding)
        self.processed_descriptions_file_path = os.path.join(self.workspace_dir, index_replaced_file)
        self.index_file = self.dataset_cfg["data"]["googlecc"]["index_file"]
        self.dataset_path = self.dataset_cfg["data"]["googlecc"]["dataset_path"]
        self.cfg = cfg
        self.unique_id = unique_id

        # Prepare the dataset processing original descriptions to new descriptions
        self.prepare_dataset()
        self.train_test_split()

        tokenize_descriptions(self.processed_descriptions_file_path, self.tokenized_descriptions_file_path)
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
        return GoogleDataGenerator(self.cfg, "train",
                                   self.vector_encoding_file_path,
                                   self.calculate_image_encoding_file_path(unique_id, "train"),
                                   self.word_dictionary_file_path), \
               GoogleDataGenerator(self.cfg, "validation",
                                   self.vector_encoding_file_path,
                                   self.calculate_image_encoding_file_path(unique_id, "validation"),
                                   self.word_dictionary_file_path), \
               GoogleDataGenerator(self.cfg, "test",
                                   self.vector_encoding_file_path,
                                   self.calculate_image_encoding_file_path(unique_id, "test"),
                                   self.word_dictionary_file_path)

    def prepare_dataset(self):
        if os.path.exists(self.processed_descriptions_file_path):
            print("Parsed descriptions exists. Reusing....")
            return

        indexes = dict()
        with open(self.index_file, 'r') as file:
            for line in file:
                values = line.strip().split(" ")
                indexes[values[0]] = values[1]

        index_token_file = open(self.processed_descriptions_file_path, 'a')
        with open(self.dataset_cfg["data"]["googlecc"]["descriptions"]) as description_file:
            for line in description_file:
                values = line.strip().split("\t")

                if values[-1] in indexes.keys():
                    values.insert(0, indexes[values[-1]]+".jpg")
                    index_token_file.write(" ".join(values[:-1]) + "\n")

    def run_one_time_encoding(self, model):
        self.encode_images(self.unique_id, model, "train")
        self.encode_images(self.unique_id, model, "validation")
        self.encode_images(self.unique_id, model, "test")

    def encode_images(self, unique_id, model, run_type):
        if os.path.exists(self.calculate_image_encoding_file_path(unique_id, run_type)):
            print("Image encodings found for {}:{}".format(unique_id, run_type))
            return

        data_list = retrieve_data_list_file(run_type)

        model = encoder_model(model)
        enc_train = model.predict_generator(self.image_generator(data_list), steps=get_line_count(data_list), verbose=1)
        print("Number of encodings in {}: {}".format(run_type, str(len(enc_train))))
        np.save(self.calculate_image_encoding_file_path(unique_id, run_type), enc_train)

    def image_generator(self, data_list):
        with open(data_list, 'r') as file:
            for line in file:
                image = cv2.imread(os.path.join(self.dataset_path, line.strip()))
                image = np.expand_dims(np.asarray(cv2.resize(image, (299, 299))) / 255.0, axis=0)
                yield image

    def calculate_image_encoding_file_path(self, unique_id, run_type):
        return os.path.join(self.workspace_dir, unique_id + "_" + run_type + "_encoding.npy")

    def train_test_split(self):

        train_split = 0.8
        validation_split = 0.1

        indexes = list()
        with open(self.index_file, 'r') as file:
            for line in file:
                values = line.strip().split(" ")
                indexes.append(values[1])

        if not os.path.exists(retrieve_data_list_file("train")):
            print("Creating train split")

            f = open(retrieve_data_list_file("train"), 'a')
            for index in indexes[:int(train_split*len(indexes))]:
                f.write(index+".jpg\n")

        if not os.path.exists(retrieve_data_list_file("validation")):
            print("Creating validation split")

            f = open(retrieve_data_list_file("validation"), 'a')
            for index in indexes[int(train_split*len(indexes)):int((train_split+validation_split) * len(indexes))]:
                f.write(index + ".jpg\n")

        if not os.path.exists(retrieve_data_list_file("test")):
            print("Creating test split")

            f = open(retrieve_data_list_file("test"), 'a')
            for index in indexes[int((train_split+validation_split) * len(indexes)):]:
                f.write(index + ".jpg\n")


class GoogleDataGenerator(Sequence):

    def __init__(self, cfg, run_type, vector_encoding_file_path, image_encoding_file_path, word_dictionary_file_path):

        if run_type == "train":
            self.batch_size = cfg["training"]["batch_size"]
        elif run_type == "validation":
            self.batch_size = cfg["validation"]["batch_size"]
        else:
            self.batch_size = 1

        self.word_to_vec_map = read_encoded_descriptions(vector_encoding_file_path)
        self.data_list = retrieve_data_list_file(run_type)
        self.image_encoding = np.load(image_encoding_file_path)
        self.previous_line_number = 0
        self.max_sentence_len = 40
        self.vocab_size = get_line_count(word_dictionary_file_path)

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

                try:
                    sequence = self.word_to_vec_map[line.strip()]
                except:
                    continue

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