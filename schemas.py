from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.api.schemas import TransactionInput, PredictionResult
from src.api.predict import predict_transaction
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Phát hiện gian lận trong giao dịch thẻ tín dụng",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResult)
async def predict(transaction: TransactionInput):
    logger.info(f"Predict request — Amount: {transaction.Amount}")
    result = predict_transaction(transaction)
    logger.info(f"Result: fraud={result.is_fraud}, prob={result.fraud_probability}")
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "service": "fraud-detection"}
