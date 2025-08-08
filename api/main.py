# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd


# Load registered model
MODEL_NAME = "CaliforniaHousingBestModel"
MODEL_STAGE = "None"  # Or use "Staging" / "Production" if registered with stage
model = mlflow.pyfunc.load_model("models/best_model")


# Define request schema
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "California Housing Model API"}

@app.post("/predict")
def predict(data: HousingData):
    try:
        input_dict = data.dict()
        print("✅ Input received:", input_dict)

        df = pd.DataFrame([input_dict])
        print("✅ DataFrame created:", df)

        prediction = model.predict(df)
        print("✅ Prediction:", prediction)

        return {"prediction": prediction[0]}
    except Exception as e:
        print("❌ Exception occurred:", str(e))
        raise HTTPException(status_code=500, detail=str(e))