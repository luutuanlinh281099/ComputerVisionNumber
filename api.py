from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from plate_ocr import LicensePlate
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import base64
import numpy as np
from random_object_id import generate
plate_ocr = LicensePlate()

class IMGBase64(BaseModel):
    base64: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

def img_from_base64(img_base64):
    img=base64.b64decode(img_base64)
    img= np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, flags=1)
    return img

@app.post("/license-plate-file")
async def LicensePlate(file: UploadFile = File(...)):
    try :
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(img.shape)
        img_path = generate() + ".jpg"
        result, plate = plate_ocr.predict(img)
        cv2.imwrite('../DoAnTotNghiep/storage/app/public/imageplate/' + img_path, img)
        return JSONResponse(status_code=200, content={ "msg" : "ocr success", "status" : True, "text" : result, "plate" : "/storage/app/public/imageplate/" + img_path})
    except :
        return JSONResponse(status_code=201, content={ "msg" : "ocr fail", "status" : False, "text" : ""}) 


@app.post("/license-plate-base64")
async def LicensePlateBase64(img_b64: IMGBase64):
    try :
        img = img_from_base64(img_b64.base64)
        print(img.shape)
        img_path = generate() + ".jpg"
        result, plate = plate_ocr.predict(img)
        cv2.imwrite('../DoAnTotNghiep/storage/app/public/imageplate/' + img_path, img)
        return JSONResponse(status_code=200, content={ "msg" : "ocr success", "status" : True, "text" : result, "plate" : "/storage/app/public/imageplate/" + img_path})
    except :
        return JSONResponse(status_code=201, content={ "msg" : "ocr fail", "status" : False, "text" : ""}) 

