import os
import yaml
import string 

def clean_tokens(sequence):
    #To do: Handle the case where it 's get split as 2 tokens it and 's
    table = str.maketrans('', '', string.punctuation)
    sequence = [x.lower() for x in sequence]
    sequence = [x.translate(table) for x in sequence]
    sequence = [x for x in sequence if x.isalpha()]
    return list(filter(lambda a: a not in ['', ' '], sequence))

def tokenize_descriptions(input_file_path, output_file_path):
    """
    Expects the file to be in a format where each line is of the format "<file_name> tokenized caption"
    Converts this and saves an output file where each line contains "<file_name>,<START>,tokenized,caption,<END>"
    and saves it in the directory mentioned under cfg["workspace"]["directory"]
    :param input_file_path:
    :param output_file_path:
    :return:
    """
    if os.path.exists(output_file_path):
        print("Tokenized descriptions found. Will not be generated.")
        return

    print("Generating tokenized descriptions")
    f = open(output_file_path, 'a')
    with open(input_file_path, 'r') as file:
        for line in file:
            if line.strip():
                sequence = line.strip().split()
                sequence = clean_tokens(sequence)
                sequence.insert(1, '<START>')
                sequence.append('<END>')
                f.write(",".join(sequence) + "\n")
    f.close()
    print("Finished generating tokenized descriptions")


def create_word_map(tokenized_descriptions_file_path, word_dictionary_output_path):
    """
    Word Map takes tokenized descriptions as input and converts it into a series of tokens and assigns an id to it.
    :param tokenized_descriptions_file_path: Path of tokenized_descriptions file
    :param word_dictionary_output_path: Path of word_dictionary file
    :return:
    """
    if os.path.exists(word_dictionary_output_path):
        print("Word map already exists in workspace. Will be reused.")
        return

    print("Word map not found. Generating....")

    words_list = []
    words_to_id = {}

    with open(tokenized_descriptions_file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split(",")
            words_list.extend(tokens[1:])

    # remove duplicate words
    words_list = list(set(words_list))

    # sorting the words
    words_list = sorted(words_list)
    for i in range(len(words_list)):
        words_to_id[words_list[i]] = i

    with open(word_dictionary_output_path, 'w') as f:
        [f.write('{0},{1}'.format(key, value) + "\n") for key, value in words_to_id.items()]


def read_word_dictionary(word_dictionary_file_path):
    """
    Read word dictionary takes a file input and assumes each line is of the following format: "word,id".
    It then converts this into a dictionary and returns
    :param word_dictionary_file_path: File path of word_dictionary file
    :return: Dictionary of words and associated ids
    """
    word_dict = dict()
    with open(word_dictionary_file_path) as f:
        for line in f:
            words = line.strip().split(",")
            word_dict[words[0]] = int(words[1])

    return word_dict


def read_id_to_word_dictionary(word_dictionary_file_path):
    """
    Read id to word dictionary takes a file input and assumes each line is of the following format: "word:id".
    It then converts this into a dictionary where key is the id and value is the word.
    :param word_dictionary_file_path: File path of word_dictionary file
    :return: Dictionary of id and associated words
    """
    word_dict = dict()
    with open(word_dictionary_file_path) as f:
        for line in f:
            words = line.strip().split(",")
            word_dict[int(words[1])] = words[0]

    return word_dict


def vector_encode_descriptions(tokenized_descriptions_file_path, vector_encoding_file_path, word_map):
    if os.path.exists(vector_encoding_file_path):
        print("Vector encoding found. Will not be generated.")
        return

    print("Generating word to vector encoding")

    #restricted_word_set = restricted_set(word_map)

    f = open(vector_encoding_file_path, 'a')
    with open(tokenized_descriptions_file_path, 'r') as file:
        for line in file:
            if line.strip():
                sequences = line.strip().split(",")
                vector_sequence = list()
                vector_sequence.append(sequences[0])

                for word in sequences[1:]:
                    #if word not in restricted_word_set:
                    vector_sequence.append(str(word_map[word]))

                f.write(",".join(vector_sequence) + "\n")
    f.close()
    print("Finished generating word to vector encoding")


# def restricted_set(word_map, threshold_frequency=5):
#     restricted_word_set = set()
#     restricted_word_set.add('')

#     return restricted_word_set


def read_encoded_descriptions(vector_encoding_file_path):
    vector_encoding_map = dict()
    with open(vector_encoding_file_path) as f:
        for line in f:
            elements = line.strip().split(",")
            vector_encoding_map[elements[0]] = elements[1:]

    return vector_encoding_map


def get_dataset_metadata_cfg():
    dataset_metadata = "../configs/dataset_metadata.yaml"

    with open(dataset_metadata) as fp:
        dataset_cfg = yaml.load(fp)

    return dataset_cfg
