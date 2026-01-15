from fastapi import FastAPI
from src.routers import predict, health, sample
from src.utils.startup import download_data, download_training_object, load_data, load_training_object 

app = FastAPI(title="GLM Logistic Regression API")

app.state.model = None
app.state.sample_data = None

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(sample.router)


@app.on_event("startup")
async def startup_event():
    """Evento de inicio para descargar y cargar el objeto training object y datos de muestra en memoria."""
    await download_training_object()
    app.state.training_object = await load_training_object()
    await download_data()
    app.state.sample_data = await load_data()
