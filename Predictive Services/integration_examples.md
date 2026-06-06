# Fusion Engine & Predictive Modeling Service Integration

This document outlines deployment notes, integration strategies, OpenAPI contracts, and standard payloads to connect the **Predictive Modeling Service** to the central **Fusion Engine**.

---

## 1. Fusion Engine Integration Example (Python)

A production-style client implementation using `httpx` to query predicted risk scores asynchronously:

```python
import httpx
import asyncio
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FusionEngineClient")

class PredictiveServiceClient:
    """Client for communicating with the Predictive Modeling Service."""
    
    def __init__(self, base_url: str, timeout: float = 2.0):
        self.base_url = base_url
        self.timeout = timeout

    async def get_hazard_prediction(
        self, 
        latitude: float, 
        longitude: float, 
        hazard_type: str,
        correlation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Sends coordinates and hazard type to prediction service and retrieves risk.
        """
        headers = {}
        if correlation_id:
            headers["X-Request-ID"] = correlation_id
            
        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": "2026-05-29T14:00:00Z",
            "hazard_type": hazard_type
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                url = f"{self.base_url}/api/v1/predict"
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Prediction failed with status: {response.status_code}. Response: {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Error connecting to predictive service: {str(e)}")
                return None

# Quick Run Demonstration
async def main():
    client = PredictiveServiceClient(base_url="http://localhost:8000")
    result = await client.get_hazard_prediction(
        latitude=12.9716, 
        longitude=77.5946, 
        hazard_type="flood",
        correlation_id="fusion-engine-uuid-1234"
    )
    print("Inference Response:", result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 2. Sample Request/Response Payloads

### Request Payload (`POST /api/v1/predict`)
```json
{
  "latitude": 12.9716,
  "longitude": 77.5946,
  "timestamp": "2026-05-29T14:00:00Z",
  "hazard_type": "flood"
}
```

### Response Payload (JSON)
```json
{
  "prediction_id": "8c5b1ae8-2023-455b-80df-8926bb0c8412",
  "predicted_risk": 0.82,
  "trend": "increasing",
  "confidence": 0.91,
  "model_version": "RandomForest",
  "processed_at": "2026-05-29T09:08:44.201Z",
  "explain": {
    "top_features": [
      {
        "feature": "rainfall_24h",
        "importance": 0.354,
        "value": 45.2
      },
      {
        "feature": "soil_moisture",
        "importance": 0.221,
        "value": 88.5
      }
    ]
  }
}
```

---

## 3. OpenAPI Schemas

The relevant JSON schemas exported for OpenAPI documentation integration:

```json
{
  "PredictionRequest": {
    "type": "object",
    "required": ["latitude", "longitude", "timestamp", "hazard_type"],
    "properties": {
      "latitude": {
        "type": "number",
        "minimum": -90.0,
        "maximum": 90.0,
        "title": "Latitude"
      },
      "longitude": {
        "type": "number",
        "minimum": -180.0,
        "maximum": 180.0,
        "title": "Longitude"
      },
      "timestamp": {
        "type": "string",
        "format": "date-time",
        "title": "Timestamp"
      },
      "hazard_type": {
        "type": "string",
        "enum": ["flood", "earthquake", "cyclone", "wildfire", "drought", "tsunami", "landslide"],
        "title": "Hazard Type"
      }
    }
  }
}
```

---

## 4. Deployment Notes

### Environment Configurations
Define standard environmental configurations in production under `.env`:

```env
ENV=production
LOG_LEVEL=info
METRICS_ENABLED=true
REDIS_URL=redis://redis:6379/0
```

### Docker Deployments
Expose the FastAPI app on port `8000` via Uvicorn in the Docker container:

```bash
docker build -t predictive-service .
docker run -p 8000:8000 --env-file .env predictive-service
```

---

## 5. cURL Demonstration Commands

### Health Endpoint Check
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

### Risk Prediction Request
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -H "X-Request-ID: FE-Task-01" \
     -d '{
       "latitude": 34.0522,
       "longitude": -118.2437,
       "timestamp": "2026-05-29T12:00:00Z",
       "hazard_type": "wildfire"
     }'
```

### Prometheus Metrics Scrape
```bash
curl -X GET "http://localhost:8000/metrics"
```
