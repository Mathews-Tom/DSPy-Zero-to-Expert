#!/usr/bin/env python3
"""
Sample DSPy Web API Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import dspy
import logging
from datetime import datetime
import psutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DSPy Sample API",
    description="A sample DSPy application for deployment demonstration",
    version="1.0.0"
)

# Configure DSPy (you would normally load this from environment)
try:
    # This would be configured with actual API keys in production
    logger.info("DSPy configuration would be loaded here")
except Exception as e:
    logger.warning(f"DSPy configuration failed: {e}")

# Sample DSPy signature
class TextAnalysisSignature(dspy.Signature):
    """Analyze text and provide insights"""
    text = dspy.InputField(desc="Text to analyze")
    sentiment = dspy.OutputField(desc="Sentiment: positive, negative, or neutral")
    summary = dspy.OutputField(desc="Brief summary of the text")

# Initialize DSPy module (with fallback)
try:
    text_analyzer = dspy.ChainOfThought(TextAnalysisSignature)
except Exception as e:
    logger.warning(f"DSPy module initialization failed: {e}")
    text_analyzer = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DSPy Sample API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "available_memory_gb": memory.available / (1024**3)
            },
            "dspy_configured": text_analyzer is not None
        }
        
        # Determine overall health
        if cpu_percent > 90 or memory.percent > 90:
            health_status["status"] = "degraded"
        
        return JSONResponse(
            status_code=200 if health_status["status"] == "healthy" else 503,
            content=health_status
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        readiness_status = {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "dspy_module": "available" if text_analyzer else "unavailable",
                "api": "ready"
            }
        }
        
        return JSONResponse(status_code=200, content=readiness_status)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/analyze")
async def analyze_text(request: dict):
    """Analyze text using DSPy"""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if text_analyzer:
            try:
                # Use DSPy for analysis
                result = text_analyzer(text=text)
                return {
                    "text": text,
                    "sentiment": result.sentiment,
                    "summary": result.summary,
                    "method": "dspy",
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.warning(f"DSPy analysis failed: {e}")
        
        # Fallback analysis
        word_count = len(text.split())
        sentiment = "neutral"  # Simple fallback
        summary = text[:100] + "..." if len(text) > 100 else text
        
        return {
            "text": text,
            "sentiment": sentiment,
            "summary": summary,
            "word_count": word_count,
            "method": "fallback",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
