# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:45:58 2019

@author: belov
"""
import datasets.flickr
from datasets.flickr import get_line_count
import argparse
import os
import sys
sys.path.append("..")
import language.beam_decoder
import yaml
from keras.models import load_model

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

dataset_preprocessor = datasets.flickr.PreProcessing(cfg)

length_test_file = get_line_count(cfg["data"]["flickr"]["test_images_file"])

model = load_model(os.path.join(cfg["workspace"]["directory"], cfg["model"]["arch"]+"_model.h5"))

training_generator, validation_generator, test_generator = dataset_preprocessor.get_keras_generators("inception")

length_test_file = get_line_count(cfg["data"]["flickr"]["test_images_file"])

for i in range(0, length_test_file):
    x, y = test_generator[i]
    sentences = language.beam_decoder.beam_search_decoder(
                    model,
                    x[0][0],
                    dataset_preprocessor.get_word_dictionary(),
                    dataset_preprocessor.get_id_dictionary(),
                    40)
    print(sentences)