import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# 1. 数据加载
train_dir = r'D:\final\train_A_cicids2017'
test_dir = r'D:\final\test_A_cicids2017'
img_size = (224, 224)
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
NUM_CLASSES = train_generator.num_classes
print('类别映射:', train_generator.class_indices)

# 2. 加载旧模型并替换输出层
old_model = load_model(r'D:\final\models_ciciot2023_h5/vgg16.h5')
x = old_model.layers[-2].output
output = Dense(NUM_CLASSES, activation='softmax', name='new_predictions')(x)
model = Model(inputs=old_model.input, outputs=output)

# 3. 冻结除新输出层以外的所有层
for layer in old_model.layers:
    layer.trainable = False
model.layers[-1].trainable = True

# 4. 编译模型
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
best_model_path = r'D:\final\models_cicids2017_h5\VGG16_new.h5'
checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
# 提前终止
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. 第一阶段：只训练输出层
history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)

# 6. 第二阶段：解冻主干最后两三个block（比如VGG16后8~10层），结合更小的学习率微调
for layer in old_model.layers[-10:]:    # 你可根据模型结构summary适当调整数量，建议先解冻最后8-12层尝试
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
# 学习率调度
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

history2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# 合并history方便画图
def merge_hist(hist1, hist2):
    h = dict(hist1.history)
    for k in hist2.history:
        if k in h: h[k] += hist2.history[k]
        else: h[k] = hist2.history[k]
    return h

full_history = merge_hist(history1, history2)

# 7. 画图
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(full_history['accuracy'], label='Train Accuracy')
plt.plot(full_history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(full_history['loss'], label='Train Loss')
plt.plot(full_history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('result_img.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. 测试表现
# 重新加载最优权重
model.load_weights(best_model_path)
test_loss, test_acc = model.evaluate(test_generator)
print(f"测试集损失: {test_loss:.4f}, 准确度: {test_acc:.4f}")