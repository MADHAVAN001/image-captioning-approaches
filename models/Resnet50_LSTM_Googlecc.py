import argparse
import os
import sys

sys.path.append("..")

import yaml
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import ResNet50
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
        default="../configs/resnet50_lstm_no_threshold.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    dataset_cfg = get_dataset_metadata_cfg()
    model_workspace_dir = os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], cfg["model"]["arch"])
    utils.make_directories(model_workspace_dir)

    img_model = ResNet50(weights='imagenet')

    dataset_preprocessor = PreProcessing(cfg, "resnet50", True, False)
    dataset_preprocessor.run_one_time_encoding(img_model)

    # Load train, validation sets from the pre-processor
    training_generator, validation_generator, test_generator = dataset_preprocessor.get_keras_generators("resnet50")

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
    sent1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(text_input)
    sent2 = Dropout(0.5)(sent1)
    sent3 = LSTM(256)(sent2)

    decoder1 = Add()([im2, sent3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    pred = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[image_input, text_input], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

    model.summary()

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(
            os.path.join(os.path.join(model_workspace_dir, 'weights_best.hdf5')),
            verbose=1,
            save_best_only=False,
            save_weights_only=True
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
