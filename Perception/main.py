from ultralytics import YOLO
import cv2

model = YOLO('YOLOv11s-Carmaker.pt')

def dist(h, H=30, f=100):
    return (H * f / h) / 100

results = model.predict(source='testing/image.webp', conf=0.5)

result = results[0]

img = cv2.imread('testing/image.webp')

color_map = {
    'Blue': (255, 0, 0),      
    'Yellow': (0, 255, 255),  
    'Small Orange': (0, 140, 255)
}

cones = []

for i, box in enumerate(result.boxes):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    h = y2 - y1  
    
    class_id = int(box.cls)
    class_name = model.names[class_id]
    
    d = dist(h)
    
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    text = f"{class_name}:{d:.2f}m"
    text_color = color_map.get(class_name, (0,0,0))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    text_x = int(x1)
    text_y = int(y1) - 5
    
    cv2.rectangle(img, (text_x, text_y-text_height-baseline),(text_x + text_width, text_y + baseline), (255,255,255), -1)
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    cones.append(text)

cv2.imwrite('annotated_output.jpg', img)

print("Detected Cones:")
for cone in cones:
    print(cone)
