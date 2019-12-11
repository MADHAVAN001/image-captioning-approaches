from tensorflow.keras import Model


def encoder_model(model):
    new_input = model.input
    new_output = model.layers[-2].output
    img_encoder = Model(new_input, new_output)

    return img_encoder
