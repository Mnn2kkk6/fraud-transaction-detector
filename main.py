"""
Entry point — chạy FastAPI server.

Cách dùng:
    python main.py
    # hoặc
    uvicorn src.api.main:app --reload --port 8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
