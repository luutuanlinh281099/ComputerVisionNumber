from .detect_num.detect_num import num_ocr
from .detect_char.detect_char import char_ocr
import torch
from torchvision import datasets, models, transforms
from PIL import Image
from .load_data import *
import cv2
class OCR:
    def __init__(self):
        self.num=num_ocr()
        self.char=char_ocr()
       
        self.input_size=(119,56)
        self.transform =  transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def taken(self,elem):
     return elem['center'][1]
    
    def taken1(self,elem):
     return elem['center'][0]
    
    def Sort(self,res):
      res.sort(key=self.taken)
      dong1=[]
      try:
        dong1.append(res[0])
      except:
        dong1=[]
      dong2=[]
      for i in range(1,len(res)-1):
        height=res[i]['height']
        if( (height/3+1)<res[i+1]['center'][1]-res[i]['center'][1]):
          dong1.append(res[i])
          for j in range(i+1,len(res)):
            dong2.append(res[j])
          break
        else: dong1.append(res[i])
      if(dong2==[] and len(dong1)>0) :dong1.append(res[-1])
      dong1.sort(key=self.taken1)
      dong2.sort(key=self.taken1)
      result={"top":dong1,"bottom":dong2}
      # for x in dong1:
      #     result.append(x)
      # for x in dong2:
      #     result.append(x)
      return result
    
    def predict(self,ims,boxes):
        # try:
            if(len(ims)>0):
                texts=""
                #for i in range(len(ims)):
                 #  ims[i]=cv2.resize(ims[i],(32,32)) 
                res=[]
                for box,img in zip(boxes,ims):
                    res.append({'center':(int((box[0]+box[2])/2),(box[1]+box[3])/2),'image':img,'height':box[3]-box[1]})
                result=self.Sort(res)
                if(len(result['bottom'])==0):
                  result=result['top']
                  list_images=[Image.fromarray(cv2.cvtColor(x['image'], cv2.COLOR_BGR2RGB)) for x in result]
                  list_torch = [self.transform(x) for x in list_images]
                  inputs=torch.stack(list_torch)
                  # num=torch.cat((inputs[0:2], inputs[3:]))
                  char=self.char.predict(inputs[2:3])
                  if(char == 'L'):
                      char_2,prob=self.char.predict_single(inputs[3:4])
                  if((char=='L' and char_2=="D") and prob>0.7):
                    num=torch.cat((inputs[0:2], inputs[4:]))
                    texts+=self.num.predict(num)
                    texts=texts[:2]+"-LD"+texts[2:]
                  else :
                    num=torch.cat((inputs[0:2], inputs[3:]))
                    texts+=self.num.predict(num)
                    texts=texts[:2]+char+"-"+texts[2:]
                  
                  return texts

                elif(len(result['top'])==3):
                  list_images=[Image.fromarray(cv2.cvtColor(x['image'], cv2.COLOR_BGR2RGB)) for x in result["top"]]
                  list_images+=[Image.fromarray(cv2.cvtColor(x['image'], cv2.COLOR_BGR2RGB)) for x in result["bottom"]]
                  list_torch = [self.transform(x) for x in list_images]
                  inputs=torch.stack(list_torch)
                  
                  num=torch.cat((inputs[0:2], inputs[3:]))
                  char=inputs[2:3]
                  texts=self.num.predict(num)
                  texts=texts[:2]+self.char.predict(char)+"-"+texts[2:]
                  return texts
                else:
                  list_images=[Image.fromarray(cv2.cvtColor(x['image'], cv2.COLOR_BGR2RGB)) for x in result["top"]]
                  list_images+=[Image.fromarray(cv2.cvtColor(x['image'], cv2.COLOR_BGR2RGB)) for x in result["bottom"]]
                  list_torch = [self.transform(x) for x in list_images]
                  inputs=torch.stack(list_torch)
                  char=inputs[2:3]
                  char=self.char.predict(char)
                  if(char!='M' and char!="A" and char!="L"):
                      num=torch.cat((inputs[0:2], inputs[3:]))
                      texts=self.num.predict(num)
                      texts=texts[:2]+"-"+char+texts[2:]
                      return texts
                  else:
                    char_2,prob=self.char.predict_single(inputs[3:4])
                    
                    if(((char=='M' and char_2=="D") or (char=='A' and char_2=="A") or (char=='L' and char_2=="D")) and prob>0.7):
                    
                      num=torch.cat((inputs[0:2], inputs[4:]))
                      texts=self.num.predict(num)
                      texts=texts[:2]+"-"+char+char_2+texts[2:]
                      return texts
                    else:
                      num=torch.cat((inputs[0:2], inputs[3:]))
                      texts=self.num.predict(num)
                      texts=texts[:2]+"-"+char+texts[2:]
                      return texts               
                # texts=self.ocr.predict(inputs)
                return texts
            else :
                
                return ""
        # except Exception as e:
        #     print(e)
        #     return ""
        
