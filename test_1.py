import torch
import cv2
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = cv2.imread('Untitled.jpeg')
results = model(img)
results.save()

print(img.shape[:2])

result = results.pandas().xyxy[0].to_numpy()
result = [item for item in result if item[6]=='person']

tmp_img1 = cv2.imread('Untitled.jpeg')
tmp_img2 = cv2.imread('Untitled.jpeg')


for i in range(len(result)):
    cv2.rectangle(tmp_img1, (int(results.xyxy[0][i][0].item()), int(results.xyxy[0][i][1].item())), (int(results.xyxy[0][i][2].item()), int(results.xyxy[0][i][3].item())), (255,255,255))
    cropped = tmp_img2[int(result[i][1]):int(result[i][3]), int(result[i][0]):int(result[i][2])]
    cv2.imwrite(f'people{i+1}.png', cropped)
cv2.imwrite('resul1.png', tmp_img1)



