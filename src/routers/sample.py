from fastapi import APIRouter, Request
from src.services.sampler_service import get_random_sample

router = APIRouter(prefix="/sample", tags=["Sample"])


@router.get("/")
def sample(request: Request):
    """
    Endpoint para obtener una muestra aleatoria de los datos de muestra.

    Args:
        request: Objeto de solicitud FastAPI

    Returns:
        dict: Muestra aleatoria de los datos de muestra
    """
    sample_data = request.app.state.sample_data
    return get_random_sample(sample_data)