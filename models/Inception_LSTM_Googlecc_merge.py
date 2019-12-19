import argparse
import os
import sys

sys.path.append("..")

import yaml
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Dropout, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from datasets.googlecc import PreProcessing, get_line_count
from datasets.common import get_dataset_metadata_cfg
from preprocessing import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.backend import clear_session
from utils.performance import PerformanceMetrics


if __name__ == "__main__":
    clear_session()
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/inception_lstm_no_threshold.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    dataset_cfg = get_dataset_metadata_cfg()
    model_workspace_dir = os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], cfg["model"]["arch"])
    utils.make_directories(model_workspace_dir)

    img_model = InceptionV3(weights='imagenet')

    dataset_preprocessor = PreProcessing(cfg, "inception", False, False)
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
    im2 = Dense(256, activation='relu')(im1)

    text_input = Input(shape=(MAX_LEN,))
    sent1 = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
    sent3 = Bidirectional(LSTM(128, return_sequences=False))(sent1)

    decoder1 = Add()([im2, sent3])
    pred = Dense(vocab_size, activation='softmax')(decoder1)

    model = Model(inputs=[image_input, text_input], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

    model.summary()

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(
            os.path.join(os.path.join(model_workspace_dir, 'weights_best.hdf5')),
            verbose=1,
            save_best_only=False
        ),
        CSVLogger(os.path.join(model_workspace_dir, 'training.csv')),
        PerformanceMetrics(os.path.join(model_workspace_dir, 'performance.csv')),
    ]

    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=100,
        callbacks=callbacks
    )
