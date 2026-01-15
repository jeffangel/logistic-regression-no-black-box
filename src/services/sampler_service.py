import numpy as np


def get_random_sample(sample_data):
    """
    Servicio que obtiene una muestra aleatoria de los datos de muestra.

    Args:
        dict: Datos de muestra y sus etiquetas cargado en memoria

    Returns:
        dict: Muestra aleatoria de los datos de muestra
    """
    
    X = sample_data["X"]
    y = sample_data["y"]
    idx = np.random.randint(0, len(X))

    data = X[idx]
    class_sample = y[idx]

    return {
        "sample": {
            "data": data.tolist(),
            "class": int(class_sample),
        }
    }
