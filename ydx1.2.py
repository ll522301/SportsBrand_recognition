# 导入必要的模块
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os, PIL, pathlib
from sklearn.metrics import classification_report
import seaborn as sns
from tensorflow.keras import regularizers
import random

# 设置随机种子
random_seed = 123
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# 加载数据集
batch_size = 16
img_height = 112
img_width = 112
data_dir = "./46-data/"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*/*.jpg')))
print("图片总数为：", image_count)

# 加载训练集和测试集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./46-data/train/",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./46-data/test/",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 获取类别名称
class_names = train_ds.class_names
print(class_names)

# 可视化训练集中的一批数据
plt.figure(figsize=(20, 5))
for images, labels in train_ds.take(1):
    for i in range(16):
        plt.subplot(2, 8, i + 1)
        plt.imshow(images[i] / 255)  # 将像素值缩放到[0,1]
        plt.title(class_names[labels[i]])
        plt.axis("off")

# cache()和prefetch()函数对数据集进行预处理和缓存以加快训练速度
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 定义卷积神经网络模型
# 该模型共包括七个层次，包括一个像素值缩放层、三个卷积层、三个池化层、两个丢弃层、一个压平层和两个全连接层。
# 其中，像素值缩放层用于将图像像素值缩放到[0,1]之间；卷积层和池化层用于提取图像的特征；丢弃层用于防止过拟合；
# 压平层用于将多维的输入数据转化为一维向量；全连接层用于对特征进行分类。
model = models.Sequential([
    # 像素值缩放层,将像素值缩放到[0,1]之间。
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    # 第一层卷积层,使用了32个大小为3x3的卷积核，relu激活函数，same padding，加入L2正则化项。
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    # 第一个池化层,使用了默认大小为2x2的窗口。
    layers.MaxPooling2D(),
    # 第一个丢弃层,丢弃15%的神经元。
    layers.Dropout(0.15),
    # 第二层卷积层,使用了64个大小为3x3的卷积核，relu激活函数，same padding，加入L2正则化项。
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    # 第二个池化层,使用了默认大小为2x2的窗口。
    layers.MaxPooling2D(),
    # 第三层卷积层,使用了128个大小为3x3的卷积核，relu激活函数，same padding，加入L2正则化项。
    layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    # 第三个池化层,使用了默认大小为2x2的窗口。
    layers.MaxPooling2D(),
    # 第二个丢弃层,丢弃15%的神经元。
    layers.Dropout(0.15),
    # 压平层,将多维的输入数据转化为一维向量。
    layers.Flatten(),
    # 第一个全连接层，512个神经元，relu激活函数，加入L2正则化项。
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    # 输出层，输出层，节点数为类别数。
    layers.Dense(len(class_names))
])


# 打印模型的结构
model.summary()

# compile()函数设置优化器、损失函数和评估指标
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# 定义一个回调函数来保存最佳模型
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./best_model.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# 定义一个学习率衰减函数,用于在训练过程中逐步降低学习率，从而使模型在训练后期更加稳定和准确。
# 具体来说，该函数采用了指数衰减的方式，即每经过decay_steps个epoch后，将学习率乘以一个衰减因子decay_rate。初始学习率initial_learning_rate为0.00001。
def lr_decay(epoch):
    initial_learning_rate = 0.00001
    decay_rate = 0.999
    decay_steps = 2
    if epoch < decay_steps:
        return initial_learning_rate
    else:
        return initial_learning_rate * decay_rate ** (epoch // decay_steps)

# 设置一个学习率调度器
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay)

# fit()函数训练模型，并添加学习率调度器和模型保存回调函数
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[lr_scheduler, model_checkpoint_callback])

# 加载最佳模型
best_model = tf.keras.models.load_model('./best_model.h5')

# np.argmax()函数对测试集进行预测
predictions = best_model.predict(val_ds)
predicted_classes = np.argmax(predictions, axis=1)

# 将val_ds转换为numpy数组，并提取标签信息
val_images = []
val_labels = []
for image, label in val_ds.unbatch().as_numpy_iterator():
    val_images.append(image)
    val_labels.append(label)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

# 计算混淆矩阵
confusion_mtx = tf.math.confusion_matrix(val_labels, predicted_classes)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names,
            annot=True, fmt='g')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 使用sklearn库中的classification_report函数计算指标
report = classification_report(val_labels, predicted_classes, target_names=class_names)
print(report)

# 可视化训练过程中的准确率和损失值
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.legend(loc='lower right')
ax1.set_title('Training and Validation Accuracy')
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.legend(loc='upper right')
ax2.set_title('Training and Validation Loss')
plt.show()
