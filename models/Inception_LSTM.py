import argparse
import os
import sys
sys.path.append("..")
import language.decoder

import yaml
from keras import Model, models
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate

from datasets.flickr import tokenize_descriptions, encode_images, get_line_count, \
    FlickrDataGenerator

import datasets.flickr


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

    encode_images(cfg, "inception", img_model, "train")
    encode_images(cfg, "inception", img_model, "validation")
    encode_images(cfg, "inception", img_model, "test")

    dataset_preprocessor = datasets.flickr.PreProcessing(cfg)

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

    training_generator, validation_generator, test_generator = dataset_preprocessor.get_keras_generators("inception")

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1)

    model.save_weights(os.path.join(cfg["workspace"]["directory"], cfg["model"]["arch"]+"_model.h5"))
    print("Saved model to disk")
    
    model.load_weights(os.path.join(cfg["workspace"]["directory"], cfg["model"]["arch"]+"_model.h5"))

    length_test_file = get_line_count(cfg["data"]["flickr"]["test_images_file"])
    f = open(os.path.join(cfg["workspace"]["directory"], "test_output.txt"), 'a')
    for i in range(0, length_test_file):
        x, y = test_generator[i]
        sentence = language.decoder.greedy_decoder(
                    model,
                    x[0][0],
                    dataset_preprocessor.get_word_dictionary(),
                    dataset_preprocessor.get_id_dictionary(),
                    40)
        f.write(" ".join(sentence[1:-1]) + "\n")
