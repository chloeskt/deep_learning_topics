import os
import pickle

from autoencoder.models import Encoder, Classifier

PARAM_LIMIT = 5e6
SIZE_LIMIT_MB = 20
ACC_THRESHOLD = 0.5


def load_model(model_path):
    model_dict = pickle.load(open(model_path, "rb"))["classifier_pt1"]

    encoder = Encoder(
        model_dict["encoder_hparam"],
        model_dict["encoder_inputsize"],
        model_dict["encoder_latent_dim"],
    )
    model = Classifier(model_dict["hparams"], encoder)
    model.load_state_dict(model_dict["state_dict"])
    return model


def save_model(model, file_name, directory="models"):
    model = model.cpu()
    model_dict = {
        "classifier_pt1": {
            "state_dict": model.state_dict(),
            "hparams": model.hparams,
            "encoder_hparam": model.encoder.hparams,
            "encoder_inputsize": model.encoder.input_size,
            "encoder_latent_dim": model.encoder.latent_dim,
            "encoder_state_dict": model.encoder.state_dict(),
        }
    }
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model_dict, open(os.path.join(directory, file_name), "wb", 4))
