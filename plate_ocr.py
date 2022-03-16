from time import time
from ocr_lpn.ocr_template import OCR
import cv2
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import numpy as np

class LicensePlate:
    def __init__(self):
        
        model_plate = "models/mb1-ssd-plate-Epoch-120-Loss-1.4717974662780762.pth"
        model_kt = 'models/mb1-ssd-kt-Epoch-95-Loss-0.9465917199850082.pth'
        label_path_plate = 'models/open-images-model-plate.txt'
        label_path_kt = 'models/open-images-model-kt.txt'
        self.ocr = OCR()
        self.model_detect_plate = self.load_model_detect(model_plate, label_path_plate)
        self.model_detect_kt = self.load_model_detect(model_kt, label_path_kt)
    
    def load_model_detect(self, model_path, label_path):
        class_names = [name.strip() for name in open(label_path).readlines()]

        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
        net.load(model_path)

        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
        return predictor

    def predict(self, img):
        boxes, _, _ = self.model_detect_plate.predict(img, 1, 0.2)
        box = np.array(boxes[0]).astype(int)
        plate = img[box[1] : box[3], box[0] : box[2], :]
        kts, _, _ = self.model_detect_kt.predict(plate, 10, 0.15)
        ims = []
        for kt in kts:
            kt = np.array(kt).astype(int)
            ims.append(plate[kt[1] : kt[3], kt[0] : kt[2], :])
        
        text = self.ocr.predict(ims, kts)
        return text, plate

import time
if __name__ == "__main__":
    X = LicensePlate()
    img=cv2.imread("15.jpg")
    t1 = time.time()
    text, plate = X.predict(img)
    print(text)
    print(time.time() - t1)
    cv2.imshow("plate", plate)
    cv2.waitKey(0)
    