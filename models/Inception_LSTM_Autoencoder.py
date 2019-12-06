import argparse
import os
import sys

sys.path.append("..")

import yaml
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate

from datasets.googlecc import PreProcessing, get_line_count
from datasets.common import get_dataset_metadata_cfg
from preprocessing import utils
from keras.callbacks import ModelCheckpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/inception_lstm_autoencoder.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    dataset_cfg = get_dataset_metadata_cfg()
    model_workspace_dir = os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], cfg["model"]["arch"])
    utils.make_directories(model_workspace_dir)

    dataset_preprocessor = PreProcessing(cfg, "autoencoder", False, False)

    # Load train, validation sets from the pre-processor
    training_generator, validation_generator, test_generator = dataset_preprocessor.get_simple_keras_generators()

    MAX_LEN = 40
    EMBEDDING_DIM = 300
    IMAGE_ENC_DIM = 300
    vocab_size = get_line_count(
        os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], "word_dictionary.txt")
    )

    img_model = InceptionV3(weights='imagenet')

    new_input = img_model.input
    new_output = img_model.layers[-2].output

    img_enc = Dense(300, activation="relu")(new_output)
    images = RepeatVector(MAX_LEN)(img_enc)

    text_input = Input(shape=(MAX_LEN,))
    embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
    x = Concatenate()([images, embedding])
    y = Bidirectional(LSTM(256, return_sequences=False))(x)
    pred = Dense(vocab_size, activation='softmax')(y)
    model = Model(inputs=[new_input, text_input], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

    model.summary()

    checkpoint = ModelCheckpoint(filepath=os.path.join(model_workspace_dir, 'weights.hdf5'),
                                 verbose=1,
                                 save_best_only=True)

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=100,
                        callbacks=[checkpoint])
