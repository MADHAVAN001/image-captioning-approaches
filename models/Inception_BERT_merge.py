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
import tensorflow_hub as hub
import tensorflow as tf

if __name__ == "__main__":
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

    MAX_LEN = 40
    EMBEDDING_DIM = 300
    IMAGE_ENC_DIM = 300
    vocab_size = get_line_count(
        os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], "word_dictionary.txt")
    )

    image_input = Input(shape=(2048,), name="image_input")
    im1 = Dropout(0.5)(image_input)
    im2 = Dense(256, activation='relu')(im1)

    text_input = Input(shape=(MAX_LEN,), dtype=tf.int32,
                                       name="text_input")
    input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32,
                                   name="input_mask")
    segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32,
                                    name="segment_ids")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)
    pooled_output, sequence_output = bert_layer([text_input, input_mask, segment_ids])
    sent = tf.keras.layers.Dense(256, activation='relu')(pooled_output)

    decoder1 = add([im2, sent])
    decoder2 = Dense(256, activation='relu')(decoder1)
    pred = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[image_input, text_input, input_mask, segment_ids], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

    model.summary()

    checkpoint = ModelCheckpoint(filepath=os.path.join(model_workspace_dir, 'weights.hdf5'),
                                 verbose=1,
                                 save_best_only=True)

    csv_logger = CSVLogger(os.path.join(model_workspace_dir, 'training.log'), append=True)

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=100,
                        callbacks=[checkpoint, csv_logger])
