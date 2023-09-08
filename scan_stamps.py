from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO

model = YOLO('./weights/last.pt')

img_path = './POD1.PNG'
img = Image.open(img_path).convert('RGB')
myFont = ImageFont.truetype('./Roboto-Bold.ttf', 30)

results = model.predict(source=img_path, conf=0.30)

i=0
for det in results[0].boxes.xyxy:
    x1, y1, x2, y2 = det
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw2 = ImageDraw.Draw(img)
    draw2.text((x1+10, y1+10),"conf : "+str(int(results[0].boxes.conf[i]*100))+"%", font=myFont,fill=(255, 0, 0),outline=(0,0,0))
    i=i+1

draw = ImageDraw.Draw(img)
if results[0].boxes.xyxy.size()[0]>=2:
    draw2.text((10, 10),"Picture with two ink stamps", font=myFont,fill=(255, 0, 0),outline=(0,0,0))
else:
    draw2.text((10, 10),"Picture with less than two ink stamps", font=myFont,fill=(255, 0, 0),outline=(0,0,0))
img.show()
