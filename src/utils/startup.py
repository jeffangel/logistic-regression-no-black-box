import pickle
import os
import shutil
import asyncio
from huggingface_hub import hf_hub_download

from src.config import settings


async def download_data():
    """Descarga los datos de muestra desde una URL si no existen localmente."""
    sample_data_path = f"{settings.DATA_FOLDER}/{settings.SAMPLE_DATA_FILENAME}"
    if os.path.exists(sample_data_path):
        print("Datos de muestra ya existen en el sistema de archivos.")
    else:
        data_cache_path = await asyncio.to_thread(
            hf_hub_download,
            repo_id=settings.FILES_ORIGIN_URL,
            filename=settings.SAMPLE_DATA_FILENAME,
            token=settings.HF_TOKEN,
            cache_dir=settings.DATA_FOLDER,
        )
        shutil.copy(data_cache_path, sample_data_path)
        print(f"Datos de muestra descargados y guardados en {sample_data_path}.")


async def download_training_object():
    """Descarga el modelo desde una URL si no existe localmente."""
    training_object_path = f"{settings.DATA_FOLDER}/{settings.TRAINING_OBJECT_FILENAME}"
    if os.path.exists(training_object_path):
        print("Modelo ya existe en el sistema de archivos.")
    else:
        training_object_cache_path = await asyncio.to_thread(
            hf_hub_download,
            repo_id=settings.FILES_ORIGIN_URL,
            filename=settings.TRAINING_OBJECT_FILENAME,
            token=settings.HF_TOKEN,
            cache_dir=settings.DATA_FOLDER,
        )
        shutil.copy(
            training_object_cache_path, training_object_path
        )
        print(
            f"Modelo descargado y guardado en {training_object_path}."
        )

async def load_data():
    """Carga los datos de muestra desde el sistema de archivos."""
    sample_data = await asyncio.to_thread(
        pickle.load,
        open(f"{settings.DATA_FOLDER}/{settings.SAMPLE_DATA_FILENAME}", "rb"),
    )
    print("Datos de muestra cargados exitosamente.")
    return sample_data


async def load_training_object():
    """Carga el modelo desde el sistema de archivos."""
    training_object = await asyncio.to_thread(
        pickle.load,
        open(f"{settings.DATA_FOLDER}/{settings.TRAINING_OBJECT_FILENAME}", "rb"),
    )
    print("Modelo cargado exitosamente.")
    return training_object
