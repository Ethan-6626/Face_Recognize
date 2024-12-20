# MTCNN+FACENET实现人脸识别

## 描述

本项目分为三个部分：

- 人脸识别并判断你是谁

- 人脸对比是否为同一人

- 实时视频的人脸视频

## 数据集准备
在dataset文件下建立人脸数据库，数据库中的样本越多识别越准确。

测试图片如合照放在test目录下。

## 运行
### 安装需要的依赖
```
opencv-python
os
numpy
mtcnn
keras_facenet
scikit-learn
```



### face_recognition
人脸识别。

因验证程序完整性（测试unknown），Broky的人脸数据未导入到dataset中。

```Python
# 使用示例
dataset_path = "dataset"  # 包含不同人物文件夹的数据集路径
image_path = "test_img/Faze/img.png"  # 待识别的图片路径
```

#### 效果

![Face Recognition_screenshot_20.12.2024](https://github.com/Ethan-6626/Face_Recognize/blob/main/result/Face%20Recognition_screenshot_20.12.2024.png)




### face_compare.py

人脸对比是否为同一人。

```python
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
```



#### 效果

![Face_Compare](https://github.com/Ethan-6626/Face_Recognize/blob/main/result/Face_Compare.png)

### RealTimeFR

实时视频人脸识别。

```python
def main():
    dataset_path = "dataset"  # 替换为你的数据集路径
    face_recognition = FaceRecognition(dataset_path)
    face_recognition.run_video()
```

#### 效果
可以自行尝试。

## 对应的计算机视觉课程设计PDF

[实验5-基于深度学习的人脸识别技术.pdf](file:///pdf/实验5-基于深度学习的人脸识别技术.pdf)

