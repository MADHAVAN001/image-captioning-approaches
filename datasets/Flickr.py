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
    with open(cfg["data"]["flickr"]["descriptions"], 'r') as file:
        for line in file:
            if line.strip():
                sequence = line.strip().split()
                sequence.insert(1, '<START>')
                sequence.append('<END>')
                f.write(",".join(sequence)+"\n")
    f.close()
    print("Finished generating tokenized descriptions")


def read_word_dictionary(cfg):
    word_dict = dict()
    with open(os.path.join(cfg["workspace"]["directory"], word_dictionary)) as f:
        for line in f:
            words = line.strip().split(",")
            word_dict[words[0]] = words[1]

    return word_dict


def read_word_to_vec(cfg):
    word_dict = dict()
    with open(os.path.join(cfg["workspace"]["directory"], vector_encoding)) as f:
        for line in f:
            elements = line.strip().split(",")
            word_dict[elements[0]] = elements[1:]

    return word_dict


def word_to_vec(cfg):
    if os.path.exists(os.path.join(cfg["workspace"]["directory"], vector_encoding)):
        print("Vector encoding found. Will not be generated.")
        return

    print("Generating word to vector encoding")

    word_map = read_word_dictionary(cfg)
    f = open(os.path.join(cfg["workspace"]["directory"], vector_encoding), 'a')
    with open(os.path.join(cfg["workspace"]["directory"], tokenized_descriptions), 'r') as file:
        for line in file:
            if line.strip():
                sequences = line.strip().split(",")
                vector_sequence = list()
                vector_sequence.append(sequences[0])

                for word in sequences[1:]:
                    vector_sequence.append(word_map[word])

                f.write(",".join(vector_sequence) + "\n")
    f.close()
    print("Finished generating word to vector encoding")


def image_generator(cfg, data_list):
    with open(data_list, 'r') as file:
        for line in file:
            image = cv2.imread(os.path.join(cfg["data"]["flickr"]["dataset_path"], line.strip()))
            image = np.expand_dims(np.asarray(cv2.resize(image, (299, 299))) / 255.0, axis=0)
            yield image


def retrieve_data_list_file(cfg, run_type):
    if run_type == "train":
        data_list = cfg["data"]["flickr"]["train_images_file"]
    elif run_type is "validation":
        data_list = cfg["data"]["flickr"]["validation_images_file"]
    else:
        data_list = cfg["data"]["flickr"]["test_images_file"]

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


class FlickrDataGenerator(Sequence):

    def __init__(self, cfg, unique_id, run_type):

        if run_type == "train":
            self.batch_size = cfg["training"]["batch_size"]
        elif run_type == "validation":
            self.batch_size = cfg["validation"]["batch_size"]

        self.token_count = get_line_count(os.path.join(cfg["workspace"]["directory"], word_dictionary))
        self.word_to_vec_map = read_word_to_vec(cfg)
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
                        input_word_sequence = pad_sequences([input_word_sequence], maxlen=self.max_sentence_len)[0]
                        output_word_sequence = to_categorical([output_word_sequence], num_classes=self.vocab_size)[0]

                        batch_images.append(self.image_encoding[index])
                        batch_word_sequences.append(input_word_sequence)
                        batch_output.append(output_word_sequence)

        return [np.array(batch_images), np.array(batch_word_sequences)], np.array(batch_output)
