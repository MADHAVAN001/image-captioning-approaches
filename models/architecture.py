from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Dropout, Add
from tensorflow.keras.applications.resnet import ResNet50


class CnnLstmInject:

    def __init__(self, max_sentence_length, embedding_dimension, vocab_size):
        self.max_sentence_length = max_sentence_length
        self.embedding_dimension = embedding_dimension
        self.vocab_size = vocab_size

    def model(self):
        """
        Model definition for inject architecture --> Encoded Image is passed as an input argument to LSTM
        :return: Defined Model
        """
        img_input = Input(shape=(2048,))

        img_enc = Dense(
            300,
            activation="relu"
        )(img_input)

        images = RepeatVector(self.max_sentence_length)(img_enc)

        # Text input
        text_input = Input(shape=(self.max_sentence_length,))
        embedding = Embedding(
            self.vocab_size,
            self.embedding_dimension,
            input_length=self.max_sentence_length
        )(text_input)

        x = Concatenate()([images, embedding])
        y = Bidirectional(
            LSTM(256, return_sequences=False)
        )(x)

        pred = Dense(
            self.vocab_size,
            activation='softmax'
        )(y)

        model = Model(inputs=[img_input, text_input], outputs=pred)

        return model


class CnnLstmMerge:

    def __init__(self, max_sentence_length, embedding_dimension, vocab_size):
        self.max_sentence_length = max_sentence_length
        self.embedding_dimension = embedding_dimension
        self.vocab_size = vocab_size

    def model(self):
        image_input = Input(shape=(2048,))
        im1 = Dropout(0.5)(image_input)
        im2 = Dense(
            256,
            activation='relu'
        )(im1)

        text_input = Input(shape=(self.max_sentence_length,))

        sent1 = Embedding(
            self.vocab_size,
            self.embedding_dimension,
            input_length=self.max_sentence_length
        )(text_input)

        sent3 = Bidirectional(LSTM(128, return_sequences=False))(sent1)

        decoder1 = Add()([im2, sent3])
        pred = Dense(self.vocab_size, activation='softmax')(decoder1)

        model = Model(inputs=[image_input, text_input], outputs=pred)

        return model


class ResnetAutoEncoder:

    def __init__(self, max_sentence_length, embedding_dimension, vocab_size):
        self.max_sentence_length = max_sentence_length
        self.embedding_dimension = embedding_dimension
        self.vocab_size = vocab_size

    def model(self):
        img_model = ResNet50(weights='imagenet')

        new_input = img_model.input
        new_output = img_model.layers[-2].output

        img_enc = Dense(300, activation="relu")(new_output)

        images = RepeatVector(
            self.max_sentence_length
        )(img_enc)

        text_input = Input(shape=(self.max_sentence_length,))

        embedding = Embedding(
            self.vocab_size,
            self.embedding_dimension,
            input_length=self.max_sentence_length
        )(text_input)

        x = Concatenate()([images, embedding])
        y = Bidirectional(LSTM(256, return_sequences=False))(x)
        pred = Dense(self.vocab_size, activation='softmax')(y)

        model = Model(inputs=[new_input, text_input], outputs=pred)
        return model
