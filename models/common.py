import os

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet import ResNet50

from datasets.googlecc import PreProcessing, get_line_count
from models.architecture import CnnLstmInject, CnnLstmMerge, ResnetAutoEncoder


def preprocessing_manager(cfg):

    if cfg["dataset"]["encoding"] == "inception":

        img_model = InceptionV3(weights='imagenet')

        dataset_preprocessor = PreProcessing(cfg, "inception", (299, 299))
        dataset_preprocessor.run_one_time_encoding(img_model)

        # Load train, validation sets from the pre-processor
        return dataset_preprocessor.get_encoding_keras_generators("inception")

    elif cfg["dataset"]["encoding"] == "resnet50":

        img_model = ResNet50(weights='imagenet')

        dataset_preprocessor = PreProcessing(cfg, "resnet50", (224, 224))
        dataset_preprocessor.run_one_time_encoding(img_model)

        # Load train, validation sets from the pre-processor
        return dataset_preprocessor.get_encoding_keras_generators("resnet50")

    elif cfg["dataset"]["encoding"] == "simple":

        dataset_preprocessor = PreProcessing(cfg, None, (299, 299))
        return dataset_preprocessor.get_image_keras_generators()

    else:
        RuntimeError("Unsupported encoding type!")


def model_loader(cfg):

    vocab_size = get_line_count(
        os.path.join(cfg["workspace"]["directory"], cfg["dataset"]["name"], "word_dictionary.txt")
    )

    if cfg["model"]["arch_type"] == "cnn_lstm_inject":
        cnn_lstm_inject = CnnLstmInject(
            max_sentence_length=40,
            embedding_dimension=300,
            vocab_size=vocab_size
        )
        return cnn_lstm_inject.model()
    elif cfg["model"]["arch_type"] == "cnn_lstm_merge":
        cnn_lstm_merge = CnnLstmMerge(
            max_sentence_length=40,
            embedding_dimension=300,
            vocab_size=vocab_size
        )
        return cnn_lstm_merge.model()
    elif cfg["model"]["arch_type"] == "resnet_lstm_autoencoder":
        resnet_autoencoder = ResnetAutoEncoder(
            max_sentence_length=40,
            embedding_dimension=300,
            vocab_size=vocab_size
        )
        return resnet_autoencoder
    else:
        RuntimeError("Unsupported model arch_type!")
