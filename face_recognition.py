import tensorflow as tf
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import os
from sklearn.preprocessing import LabelEncoder
import pickle

# 初始化MTCNN和FaceNet模型
detector = MTCNN()
facenet = FaceNet()


def load_face_dataset(dataset_path):
    face_embeddings = []
    face_names = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                faces = detector.detect_faces(img)
                if faces:
                    x, y, w, h = faces[0]['box']
                    face = img[y:y + h, x:x + w]
                    face = cv2.resize(face, (160, 160))

                    face_embedding = facenet.embeddings([face])[0]

                    face_embeddings.append(face_embedding)
                    face_names.append(person_name)

    return np.array(face_embeddings), np.array(face_names)


def recognize_faces(image_path, dataset_path):
    known_embeddings, known_names = load_face_dataset(dataset_path)

    le = LabelEncoder()
    known_names_encoded = le.fit_transform(known_names)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img_rgb)

    for face in faces:
        x, y, w, h = face['box']
        face_img = img_rgb[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (160, 160))

        face_embedding = facenet.embeddings([face_img])[0]

        distances = np.linalg.norm(known_embeddings - face_embedding, axis=1)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        threshold = 0.8  # 识别阈值
        if min_distance < threshold:
            name = known_names[min_distance_idx]
            # 绿框表示已识别的人脸
            color = (0, 255, 0)
        else:
            name = "Unknown"
            # 红框表示未知人脸
            color = (0, 0, 255)

        # 绘制人脸框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # 添加名字标签
        label_size, baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_label = max(y - 10, label_size[1])

        # 绘制标签背景
        cv2.rectangle(img, (x, y_label - label_size[1]), (x + label_size[0], y_label + baseline),
                      color, cv2.FILLED)
        # 绘制标签文字
        cv2.putText(img, name, (x, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


# 使用示例
dataset_path = "dataset"  # 包含不同人物文件夹的数据集路径
image_path = "test_img/Faze/img.png"  # 待识别的图片路径

result = recognize_faces(image_path, dataset_path)
cv2.imshow("Face Recognition", result)
cv2.waitKey(0)
cv2.destroyAllWindows()