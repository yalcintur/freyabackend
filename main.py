from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uuid
import torch
import model as model
from PIL import Image
import os
from io import BytesIO


app = FastAPI()

model.initialize(path = "./model_50_2023_01_08_00_54_14.pt")


@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):

    # TODO:
    # 1. Solve tranparancy error handling problem
    
    contents = await file.read()  # <-- Important!

    if(file.content_type.split("/")[0] != "image"):
        return {"error": "File is not an image"}

    image = Image.open(BytesIO(contents))
    image.seek(0)
    prediction, confidence = model.evaluate(image)


    print(prediction, confidence)
    
    return {"prediction": prediction.item(), "confidence": confidence.item()}

@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}