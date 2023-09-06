# STAMP DETECTION


# Stamp Detection

This Python script detects stamps in an image using the YOLOv5 object detection model.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- PIL (Python Imaging Library)
- torch
- torchvision
- yolov5

You can install these dependencies using `pip` by running the following command:
```
pip install pillow torch torchvision yolov5
```

## Usage

1. Import the required libraries:

```python
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
```
Load the YOLOv5 model:
```python
model = attempt_load('yolov5s.pt', map_location=torch.device('cpu'))
```

Make sure to replace 'yolov5s.pt' with the correct path to your YOLOv5 model file.

Load and preprocess the input image:
```python
image = Image.open('input_image.jpg')
image_tensor = F.to_tensor(image)
```

## Acknowledgements
YOLOv5: https://github.com/ultralytics/yolov5
Feel free to customize or enhance the code as per your requirements.