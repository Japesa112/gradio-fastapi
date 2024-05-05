from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI()

class URLInput(BaseModel):
    url_input: str

@app.post("/predict")
def predict_url(input_data: URLInput):
    client = Client("Nyandori/whisper")
    result = client.predict(
        url_input=input_data.url_input,
        api_name="/predict"
    )
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
