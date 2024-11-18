import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """
    加载并预处理 MNIST 测试数据集。
    1. 将图像数据的形状从 [28, 28] 转换为 [28, 28, 1] 以适配模型输入。
    2. 将像素值归一化到 [0, 1]。
    3. 将标签进行 One-Hot 编码。

    Returns:
        test_images: 预处理后的测试图像数据。
        test_labels: 预处理后的测试标签数据。
    """
    # 加载 MNIST 数据集
    (_, _), (test_images, test_labels) = mnist.load_data()

    # 数据预处理
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)) / 255.0
    test_labels = to_categorical(test_labels, 10)

    return test_images, test_labels


def load_model(model_path='googlenet_mnist.h5'):
    """
    加载训练好的模型。

    Args:
        model_path: 模型保存的路径，默认为 'googlenet_mnist.h5'。

    Returns:
        model: 加载的训练好的模型。
    """
    model = tf.keras.models.load_model(model_path)
    return model


def evaluate_model(model, test_images, test_labels):
    """
    评估模型在测试集上的表现。

    Args:
        model: 已加载的训练好的模型。
        test_images: 预处理后的测试图像数据。
        test_labels: 预处理后的测试标签数据。

    Returns:
        test_loss: 测试集上的损失值。
        test_acc: 测试集上的准确率。
    """
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc


def display_predictions(model, test_images, test_labels, num_images=6):
    """
    显示前几张测试图片及其预测结果。

    Args:
        model: 已加载的训练好的模型。
        test_images: 预处理后的测试图像数据。
        test_labels: 预处理后的测试标签数据。
        num_images: 显示图片的数量，默认为 6。
    """
    # 进行预测
    predictions = model.predict(test_images[:num_images])

    # 显示前 num_images 张测试图片及其预测结果
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # 2行3列
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(test_images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Pred: {tf.argmax(predictions[i]).numpy()} \n True: {tf.argmax(test_labels[i]).numpy()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    主程序函数，加载数据、加载模型、评估模型并显示预测结果。
    """
    # 加载并预处理数据
    test_images, test_labels = load_and_preprocess_data()

    # 加载训练好的模型
    model = load_model('googlenet_mnist.h5')

    # 评估模型
    evaluate_model(model, test_images, test_labels)

    # 显示预测结果
    display_predictions(model, test_images, test_labels)


if __name__ == '__main__':
    main()
