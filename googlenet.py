import tensorflow as tf
from tensorflow.keras import layers, models


def googlenet(input_shape=(28, 28, 1), num_classes=10):
    """
    创建一个简化版的 GoogLeNet 模型，用于处理图像分类任务（如 MNIST 数据集）。

    参数:
        input_shape (tuple): 输入图像的形状，默认值为 (28, 28, 1)，适用于单通道图像（例如 MNIST）。
        num_classes (int): 分类的类别数量，默认为 10 类（适用于 MNIST）。

    返回:
        model (tf.keras.Model): 构建的 Keras 模型。
    """
    # 输入层
    inputs = layers.Input(shape=input_shape)

    # 第一卷积层 + 池化层
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # 第二卷积层 + 池化层
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 第三卷积层 + 池化层
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Inception 模块
    inception1 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    inception2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    inception3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)

    # 拼接 Inception 模块的输出
    x = layers.concatenate([inception1, inception2, inception3], axis=-1)

    # 全局平均池化层
    x = layers.GlobalAveragePooling2D()(x)

    # 全连接层
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout 层减少过拟合
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # 输出层，使用 softmax 激活函数进行多分类

    # 创建并返回模型
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
