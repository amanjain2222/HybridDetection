from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import asyncio

app = FastAPI()

# List of service endpoints
SERVICE_URLS = [
    "http://118.138.234.214:30003",
    "http://118.138.234.3:30003",
]

# Round robin index and lock
rr_index = 0
rr_lock = asyncio.Lock()


async def get_next_url():
    global rr_index
    async with rr_lock:
        url = SERVICE_URLS[rr_index]
        rr_index = (rr_index + 1) % len(SERVICE_URLS)
    return url


@app.get("/")
async def root():
    try:
        url = await get_next_url()
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/")
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict")
async def gateway_predict(request: Request):
    try:
        url = await get_next_url()
        payload = await request.json()

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{url}/predict", json=payload)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
