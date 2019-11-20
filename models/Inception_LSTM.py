import argparse
import os
import sys
import language.decoder

import yaml
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate

sys.path.append("..")

from datasets.flickr import tokenize_descriptions, create_word_map, word_to_vec, encode_images, get_line_count, \
    FlickrDataGenerator, read_word_dictionary, read_id_to_word_dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/inception_lstm.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    img_model = InceptionV3(weights='imagenet')

    tokenize_descriptions(cfg)
    create_word_map(cfg)
    word_to_vec(cfg)
    encode_images(cfg, "inception", img_model, "train")
    encode_images(cfg, "inception", img_model, "validation")
    encode_images(cfg, "inception", img_model, "test")

    MAX_LEN = 40
    EMBEDDING_DIM = 300
    IMAGE_ENC_DIM = 300
    vocab_size = get_line_count(os.path.join(cfg["workspace"]["directory"], "word_dictionary.txt"))

    img_input = Input(shape=(2048,))
    img_enc = Dense(300, activation="relu")(img_input)
    images = RepeatVector(MAX_LEN)(img_enc)

    # Text input
    text_input = Input(shape=(MAX_LEN,))
    embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
    x = Concatenate()([images, embedding])
    y = Bidirectional(LSTM(256, return_sequences=False))(x)
    pred = Dense(vocab_size, activation='softmax')(y)
    model = Model(inputs=[img_input, text_input], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

    model.summary()
    training_generator = FlickrDataGenerator(cfg, "inception", "train")
    validation_generator = FlickrDataGenerator(cfg, "inception", "validation")
    test_generator = FlickrDataGenerator(cfg, "inception", "validation")

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1)

    model.save_weights(os.path.join(cfg["workspace"]["directory"], "model.h5"))
    print("Saved model to disk")

    f = open(os.path.join(cfg["workspace"]["directory"], "test_output.txt"), 'a')
    for data in test_generator.__getitem__(0):
        sentence = language.decoder.greedy_decoder(
            model,
            encode_images,
            read_word_dictionary(cfg),
            read_id_to_word_dictionary(cfg),
            40)
        f.write("".join(sentence) + "\n")






