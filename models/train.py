import argparse
import os
import sys

import yaml

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.backend import clear_session

sys.path.append("..")

from utils.performance import PerformanceMetrics
from models.common import preprocessing_manager, model_loader
from datasets.common import get_dataset_metadata_cfg
from preprocessing import utils


def train():

    # Clear tf session
    clear_session()
    training_generator, validation_generator, _ = preprocessing_manager(cfg)

    # Load appropriate model based on cfg
    model = model_loader(cfg)

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer="Adam",
        metrics=['accuracy']
    )

    model.summary()

    # Set appropriate callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(
            os.path.join(os.path.join(model_workspace_dir, 'weights_best.hdf5')),
            verbose=1,
            save_best_only=False,
            save_weights_only=False
        ),
        CSVLogger(os.path.join(model_workspace_dir, 'training.csv')),
        PerformanceMetrics(os.path.join(model_workspace_dir, 'performance.csv')),
    ]

    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=cfg["training"]["epochs"],
        callbacks=callbacks
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/resnet_lstm_inject_no_threshold.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    dataset_cfg = get_dataset_metadata_cfg()
    model_workspace_dir = os.path.join(
        cfg["workspace"]["directory"],
        cfg["dataset"]["name"],
        cfg["model"]["name"]
    )

    # Ensure the directory exists
    utils.make_directories(model_workspace_dir)
