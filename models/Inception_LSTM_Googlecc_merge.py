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
from keras.callbacks import CSVLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/inception_lstm_preprocessed1.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    dataset_cfg = get_dataset_metadata_cfg()
    model_workspace_dir = os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], cfg["model"]["arch"])
    utils.make_directories(model_workspace_dir)

    img_model = InceptionV3(weights='imagenet')

    dataset_preprocessor = PreProcessing(cfg, "inception")
    dataset_preprocessor.run_one_time_encoding(img_model)

    # Load train, validation sets from the pre-processor
    training_generator, validation_generator, test_generator = dataset_preprocessor.get_keras_generators("inception")

    MAX_LEN = 40
    EMBEDDING_DIM = 300
    IMAGE_ENC_DIM = 300
    vocab_size = get_line_count(
        os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], "word_dictionary.txt")
    )

    image_input = Input(shape=(2048,))
    im1 = Dropout(0.5)(image_input)
    im2 = Dense(256, activation='relu')(fe1)

    text_input = Input(shape=(MAX_LEN,))
    sent1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(text_input)
    sent2 = Dropout(0.5)(sent1)
    sent3 = LSTM(256)(sent2)

    decoder1 = add([im2, sent3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    pred = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[image_input, text_input], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

    model.summary()

    checkpoint = ModelCheckpoint(filepath=os.path.join(model_workspace_dir, 'weights.hdf5'),
                                 verbose=1,
                                 save_best_only=True)

    csv_logger = CSVLogger(os.path.join(model_workspace_dir, 'training.log'), append=True)

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=100,
                        callbacks=[checkpoint, csv_logger])
