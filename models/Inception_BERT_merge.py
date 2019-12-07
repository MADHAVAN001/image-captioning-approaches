import argparse
import os
import sys

sys.path.append("..")

import yaml
from tensorflow.keras.applications.inception_v3 import InceptionV3

from datasets.googlecc import PreProcessing
from datasets.common import get_dataset_metadata_cfg
from preprocessing import utils

import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Dense, Input, Add, Dropout
from utils.performance import PerformanceMetrics

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
MAX_LEN = 40
vocab_size = 30522


def bert_model():
    image_input = Input(shape=(2048,))
    im1 = Dropout(0.5)(image_input)
    im2 = Dense(256, activation='relu')(im1)

    text_input = Input(shape=(MAX_LEN,), dtype=tf.int32)
    input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32)
    segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32)

    bert_module = hub.Module(
        BERT_MODEL_HUB,
        trainable=True
    )

    bert_inputs = dict(
        input_ids=text_input,
        input_mask=input_mask,
        segment_ids=segment_ids
    )

    pooled_output = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True
    )["pooled_output"]

    sent = Dense(256, activation='relu')(pooled_output)

    decoder1 = Add()([im2, sent])
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    pred = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    model = tf.keras.models.Model(inputs=[image_input, text_input, input_mask, segment_ids], outputs=pred)

    return model


if __name__ == "__main__":
    clear_session()
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/inception_bert.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    dataset_cfg = get_dataset_metadata_cfg()
    model_workspace_dir = os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], cfg["model"]["arch"])
    utils.make_directories(model_workspace_dir)

    img_model = InceptionV3(weights='imagenet')

    dataset_preprocessor = PreProcessing(cfg, "inception", False, True)
    dataset_preprocessor.run_one_time_encoding(img_model)

    # Load train, validation sets from the pre-processor
    training_generator, validation_generator, test_generator = dataset_preprocessor.get_keras_generators("inception")

    model = bert_model()

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

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
