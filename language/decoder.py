import numpy as np
from keras.preprocessing.sequence import pad_sequences


def greedy_decoder(model, encoded_image, word_map, id_to_word_dictionary, max_sentence_length):

    predicted_words = list()
    predicted_words.append(word_map["<START>"])

    while predicted_words[-1] is not word_map["<END>"] and len(predicted_words) < max_sentence_length:
        input_word_sequence = pad_sequences([predicted_words], maxlen=max_sentence_length)[0]
        next_word_probabilites = model.predict(
            [np.array([encoded_image]), np.array([input_word_sequence])]
        )
        next_word = np.argmax(next_word_probabilites)
        predicted_words.append(next_word)

    sentence = list()
    for id_to_word_dictionary[id] in predicted_words:
        if id_to_word_dictionary[id] is "<END>":
            break
        sentence.append(id_to_word_dictionary[id])

    return sentence
