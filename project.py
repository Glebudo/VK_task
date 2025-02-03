import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Конфигурации
LOGO_CLASSIFIER_WEIGHTS = 'logo_classifier.pth'  # Путь к весам классификатора
CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности для детекции
CLASSIFICATION_THRESHOLD = 0.9  # Порог уверенности для классификации

# Загрузка модели YOLOv5
model_yolo = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Используем YOLOv5 как пример

# Загрузка модели классификации логотипов
class LogoClassifier(torch.nn.Module):
    def __init__(self):
        super(LogoClassifier, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)  # Бинарная классификация

    def forward(self, x):
        return self.model(x)

model_classifier = LogoClassifier()
model_classifier.load_state_dict(torch.load(LOGO_CLASSIFIER_WEIGHTS))
model_classifier.eval()

# Преобразование изображения для классификации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Функция для классификации логотипа
def classify_logo(image):
    image = Image.fromarray(image)  # Преобразование в PIL Image
    image = transform(image).unsqueeze(0)  # Применение преобразований
    with torch.no_grad():
        prediction = torch.sigmoid(model_classifier(image))
    return prediction.item() > CLASSIFICATION_THRESHOLD

# Основная функция обработки изображения
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка загрузки изображения.")
        return

    # Детекция логотипов с использованием YOLOv9
    results = model_yolo(image)
    detections = results.xyxy[0].numpy()  # Получение результатов детекции

    # Обработка каждого обнаруженного логотипа
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Извлечение области с логотипом
        logo = image[int(y1):int(y2), int(x1):int(x2)]

        # Классификация логотипа
        if classify_logo(logo):
            print(f"Логотип обнаружен в области: ({x1}, {y1}, {x2}, {y2}). Это логотип искомой организации.")
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        else:
            print(f"Логотип обнаружен в области: ({x1}, {y1}, {x2}, {y2}). Это не логотип искомой организации.")
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Отображение результата
    cv2.imshow("Detected Logos", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Пример использования
process_image('test_1.jpg')
