import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from googlenet import googlenet


def load_and_preprocess_data():
    """
    加载 MNIST 数据集并进行预处理。
    1. 将数据集的图像从 [28, 28] 形状扩展为 [28, 28, 1]，以符合卷积层的输入要求。
    2. 将图像像素值归一化到 [0, 1] 范围。
    3. 将标签进行 One-Hot 编码。

    Returns:
        train_images: 预处理后的训练图像数据。
        train_labels: 预处理后的训练标签数据。
        test_images: 预处理后的测试图像数据。
        test_labels: 预处理后的测试标签数据。
    """
    # 加载 MNIST 数据集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 数据预处理
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)) / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)) / 255.0

    # 将标签进行 One-Hot 编码
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def create_model(input_shape=(28, 28, 1), num_classes=10):
    """
    创建并返回 GoogLeNet 模型。

    Args:
        input_shape: 输入数据的形状，默认为 (28, 28, 1) 适用于 MNIST 数据集。
        num_classes: 输出类别数，默认为 10，适用于 MNIST 数据集。

    Returns:
        model: 构建好的 GoogLeNet 模型。
    """
    # 创建 GoogLeNet 模型
    model = googlenet(input_shape=input_shape, num_classes=num_classes)

    # 编译模型
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, train_images, train_labels, epochs=5, batch_size=64):
    """
    训练模型。

    Args:
        model: 已编译的模型。
        train_images: 训练数据集的图像。
        train_labels: 训练数据集的标签。
        epochs: 训练的轮数，默认为 5。
        batch_size: 批次大小，默认为 64。

    Returns:
        history: 训练过程中的历史记录，包含损失和准确率。
    """
    # 训练模型
    history = model.fit(train_images, train_labels,
                        epochs=epochs,
                        batch_size=batch_size)

    return history


def save_model(model, filename='googlenet_mnist.h5'):
    """
    保存训练好的模型。

    Args:
        model: 已训练的模型。
        filename: 保存模型的文件名，默认为 'googlenet_mnist.h5'。
    """
    model.save(filename)
    print(f"Model saved to {filename}")


def main():
    """
    主程序函数，加载数据、创建和训练模型、保存模型。
    """
    # 加载并预处理数据
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    # 创建模型
    model = create_model(input_shape=(28, 28, 1), num_classes=10)

    # 训练模型
    train_model(model, train_images, train_labels, epochs=5, batch_size=64)

    # 保存模型
    save_model(model)


if __name__ == '__main__':
    main()
