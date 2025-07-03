# Animal-Detection-using-YOLOv8
 This project uses YOLOv8 to detect and classify animals in images/videos. Carnivores are highlighted in red, with a popup showing count.
# Training will create:
- animal_detection/yolov8_animals/weights/best.pt - Best model
- animal_detection/yolov8_animals/weights/last.pt - Last checkpoint
- Training plots and metrics
## Features
- Detects 20+ animal species
- Highlights carnivores (Lion, Tiger, etc.) in red
- GUI for image and video upload
- Pop-up message for carnivore count
## Carnivorous Animal Detection
- The system automatically identifies and highlights these carnivorous animals in RED:
Lion
Leopard
Cheetah
Tiger
Bear
Brown Bear
Polar Bear
Crocodile
Cat
- Non-carnivorous animals are highlighted in GREEN.

## How to Run
```bash
python gui_detect.py

## **Structure**
Animal_Detection_YOLOV8-main/
├── dataset/
│   ├── images/
│   │   ├── train/          
│   │   ├── val/           
│   │   └── test/          
│   └── labels/             
│       ├── train/          
│       ├── val/            
│       └── test/          
├── animal_data.yaml
| detect.py             
├── train_yolo.py   
