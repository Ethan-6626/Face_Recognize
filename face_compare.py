import tensorflow as tf
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

# 初始化MTCNN和FaceNet模型
detector = MTCNN()
facenet = FaceNet()


def get_face_embedding(image_path):
    """
    获取图片中人脸的特征向量
    """
    # 读取图片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测人脸
    faces = detector.detect_faces(img)
    if not faces:
        return None, None

    # 获取第一个人脸
    x, y, w, h = faces[0]['box']
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, (160, 160))

    # 获取人脸特征向量
    face_embedding = facenet.embeddings([face])[0]

    # 在原图上画框
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return face_embedding, img


def verify_faces(image_path1, image_path2, threshold=0.8):
    """
    验证两张图片中的人脸是否为同一人
    """
    # 获取两张图片的人脸特征向量
    embedding1, img1 = get_face_embedding(image_path1)
    embedding2, img2 = get_face_embedding(image_path2)

    if embedding1 is None or embedding2 is None:
        return None, None, None

    # 计算欧氏距离
    distance = np.linalg.norm(embedding1 - embedding2)

    # 判断是否为同一人
    is_same = distance < threshold

    return is_same, img1, img2


def display_result(img1, img2, is_same):
    """
    显示比较结果
    """
    plt.figure(figsize=(10, 5))

    # 显示第一张图片
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Image 1')

    # 显示第二张图片
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Image 2')

    # 显示比较结果
    if is_same:
        plt.suptitle('Same Person', color='green', size=16)
    else:
        plt.suptitle('Different Person', color='red', size=16)

    plt.show()


def main():
    # 设置两张待比较的图片路径
    image_path1 = "dataset/Donald_Trump/1.png"
    image_path2 = "test_img/t1.png"

    # 验证人脸
    is_same, img1, img2 = verify_faces(image_path1, image_path2)

    if img1 is None or img2 is None:
        print("No face detected in one or both images!")
        return

    # 显示结果
    display_result(img1, img2, is_same)


if __name__ == "__main__":
    main()