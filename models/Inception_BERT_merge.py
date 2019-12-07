import argparse
import os
import sys

sys.path.append("..")

import yaml
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Dropout, Add, Lambda

from datasets.googlecc import PreProcessing, get_line_count
from datasets.common import get_dataset_metadata_cfg
from preprocessing import utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import tensorflow_hub as hub
import tensorflow as tf
from keras.backend import clear_session
from tensorflow.keras import backend as K

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                "Undefined pooling type (must be either first or mean"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable     #, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                "Undefined pooling type (must be either first or mean"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append("encoder/layer_{}".format(str(11 - i)))

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError("Undefined pooling type (must be either first or mean")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

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

    MAX_LEN = 40
    EMBEDDING_DIM = 300
    IMAGE_ENC_DIM = 300
    vocab_size = 30522


    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    with tf.Graph().as_default():
        image_input = tf.keras.layers.Input(shape=(2048,))
        print(image_input.shape)
        im1 = tf.keras.layers.Dropout(0.5)(image_input)
        im2 = tf.keras.layers.Dense(256, activation='relu')(im1)
        print(im2.shape)

#         input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32,
#                                        name="input_ids")
#         input_mask = Input(shape=(MAX_LEN,), dtype=tf.int32,
#                                    name="input_mask")
#         segment_ids = Input(shape=(MAX_LEN,), dtype=tf.int32,
#                                     name="segment_ids")
#         bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
#         bert_inputs = dict(input_ids=input_ids,input_mask=input_mask,segment_ids=segment_ids)
#         pooled_output = bert_module(inputs=bert_inputs,signature="tokens",as_dict=True)["pooled_output"]
#         print("shape", pooled_output.shape)
#         bert_out = Lambda(convert_tensor, output_shape=(None,768))(pooled_output)
#         print("shape", bert_out.shape)
#     bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
#                             trainable=True)
#     pooled_output, sequence_output = bert_layer([text_input, input_mask, segment_ids])
        in_id = tf.keras.layers.Input(shape=(MAX_LEN,))
        in_mask = tf.keras.layers.Input(shape=(MAX_LEN,))
        in_segment = tf.keras.layers.Input(shape=(MAX_LEN,))
        bert_inputs = [in_id, in_mask, in_segment]
    
        bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)
        sent = tf.keras.layers.Dense(256, activation='relu')(bert_output)
        print(sent.shape)

        decoder1 = tf.keras.layers.Add()([im2, sent])
        decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
        pred = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

        model = tf.keras.models.Model(inputs=[image_input, in_id, in_mask, in_segment], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        print("Compiled model. OK")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_workspace_dir, 'weights.hdf5'),
                                 verbose=1,
                                 save_best_only=True)

        #csv_logger = CSVLogger(os.path.join(model_workspace_dir, 'training.log'), append=True)

        model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=1,
                        callbacks=[checkpoint])
