import tensorflow as tf
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import time
from collections import deque


class FaceRecognition:
    def __init__(self, dataset_path):
        self.detector = MTCNN()  # 移除min_face_size参数
        self.facenet = FaceNet()
        self.dataset_path = dataset_path
        self.known_embeddings = []
        self.known_names = []
        self.face_locations = {}  # 存储人脸位置
        self.face_labels = {}  # 存储人脸标签
        self.threshold = 0.8
        self.min_detection_confidence = 0.95  # 添加检测置信度阈值
        self.load_dataset()

    def load_dataset(self):
        """加载数据集并预计算特征向量"""
        print("Loading dataset...")
        for person_name in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    faces = self.detector.detect_faces(img)
                    if faces and faces[0]['confidence'] > self.min_detection_confidence:
                        x, y, w, h = faces[0]['box']
                        face = img[y:y + h, x:x + w]
                        face = cv2.resize(face, (160, 160))
                        face_embedding = self.facenet.embeddings([face])[0]

                        self.known_embeddings.append(face_embedding)
                        self.known_names.append(person_name)

        self.known_embeddings = np.array(self.known_embeddings)
        print(f"Dataset loaded: {len(self.known_names)} faces")

    def track_face(self, current_faces, frame_shape):
        """更新人脸跟踪位置"""
        height, width = frame_shape[:2]
        current_locations = {}

        # 只处理置信度高的人脸
        for face in current_faces:
            if face['confidence'] > self.min_detection_confidence:
                x, y, w, h = face['box']
                center = (x + w // 2, y + h // 2)
                current_locations[center] = (x, y, w, h)

        # 如果是第一帧或者没有历史位置，直接使用当前位置
        if not self.face_locations:
            self.face_locations = current_locations
            return current_locations

        # 更新跟踪位置
        new_locations = {}
        for center, bbox in current_locations.items():
            matched = False
            for old_center, old_bbox in self.face_locations.items():
                # 计算中心点距离
                distance = np.sqrt((center[0] - old_center[0]) ** 2 +
                                   (center[1] - old_center[1]) ** 2)
                # 如果距离小于阈值，认为是同一个人脸
                if distance < 50:  # 降低距离阈值，提高稳定性
                    new_locations[center] = bbox
                    matched = True
                    break
            if not matched:
                new_locations[center] = bbox

        self.face_locations = new_locations
        return new_locations

    def process_frame(self, frame):
        """处理单帧图像"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_frame)

        # 更新人脸跟踪位置
        face_locations = self.track_face(faces, frame.shape)

        for center, (x, y, w, h) in face_locations.items():
            # 如果该位置已经有标签且标签仍然有效，直接使用已有标签
            if center in self.face_labels and time.time() - self.face_labels[center]['time'] < 0.5:  # 降低缓存时间
                name = self.face_labels[center]['name']
                color = self.face_labels[center]['color']
            else:
                # 提取人脸区域并识别
                face_img = rgb_frame[y:y + h, x:x + w]
                try:
                    face_img = cv2.resize(face_img, (160, 160))
                    face_embedding = self.facenet.embeddings([face_img])[0]

                    distances = np.linalg.norm(self.known_embeddings - face_embedding, axis=1)
                    min_distance_idx = np.argmin(distances)
                    min_distance = distances[min_distance_idx]

                    if min_distance < self.threshold:
                        name = self.known_names[min_distance_idx]
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)

                    # 更新标签缓存
                    self.face_labels[center] = {
                        'name': name,
                        'color': color,
                        'time': time.time()
                    }
                except:
                    continue

            # 绘制人脸框和标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0], y), color, cv2.FILLED)
            cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        return frame

    def run_video(self):
        """运行实时视频识别"""
        cap = cv2.VideoCapture(0)

        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        fps_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 计算和显示FPS
            if time.time() - fps_time > 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
                print(f"FPS: {fps}")

            # 处理帧
            frame = self.process_frame(frame)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    dataset_path = "dataset"  # 替换为你的数据集路径
    face_recognition = FaceRecognition(dataset_path)
    face_recognition.run_video()


if __name__ == "__main__":
    main()