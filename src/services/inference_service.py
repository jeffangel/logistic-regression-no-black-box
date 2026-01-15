import numpy as np

from src.utils.activations import sigmoid

def predict_class(new_sample_raw, training_object):
    """
    Servicio que realiza una predicción de clase a partir de una nueva muestra.

    Args:
        new_sample_raw: Nueva muestra para la predicción
        training_object: Objeto de entrenamiento cargado en memoria

    Returns:
        dict: Diccionario con la predicción de clase
    """
    
    mean = training_object["mean"]
    std = training_object["std"]
    beta_nr = training_object["beta_nr"]
    beta_gd = training_object["beta_gd"]
    beta_bt = training_object["beta_bt"]

    new_sample_raw = np.array(new_sample_raw)
    new_sample_std = (new_sample_raw - mean) / std
    new_sample = np.hstack([1.0, new_sample_std])

    eta_nr = np.dot(new_sample, beta_nr)
    eta_gd = np.dot(new_sample, beta_gd)
    eta_bt = np.dot(new_sample, beta_bt)

    p_nr = sigmoid(eta_nr)
    p_gd = sigmoid(eta_gd)
    p_bt = sigmoid(eta_bt)

    return {
        "newton": {
            "eta": eta_nr.item(),
            "prob": p_nr.item()
        },
        "gradient_descent": {
            "eta": eta_gd.item(),
            "prob": p_gd.item()
        },
        "gradient_descent_backtracking": {
            "eta": eta_bt.item(),
            "prob": p_bt.item()
        }
    }