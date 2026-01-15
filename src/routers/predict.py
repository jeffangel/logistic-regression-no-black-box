from fastapi import APIRouter, Request, Body
from src.services.inference_service import predict_class

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/")
def predict(request: Request, new_sample: list = Body(...)):
    """
    Endpoint de predicción de edad y género a partir de una imagen.

    Args:
        request: Objeto de solicitud FastAPI
        new_sample: Nueva muestra para la predicción
        training_object: Objeto de entrenamiento cargado en memoria

    Returns:
        dict: Diccionario con la predicción de edad y género
    """
    training_object = request.app.state.training_object
    return predict_class(new_sample, training_object)
