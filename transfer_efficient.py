import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# 1. 数据加载
train_dir = r'D:\final\train_A_cicids2017'
test_dir = r'D:\final\test_A_cicids2017'
img_size = (224, 224)
batch_size = 16

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

# 2. 加载预训练EfficientNetB7模型并替换输出层
old_model = load_model(r'D:\final\models_ciciot2023_h5/efficientnetb7.h5')
x = old_model.layers[-2].output  # 最后一层前的输出（如报错用model.summary()查找）
output = Dense(NUM_CLASSES, activation='softmax', name='new_predictions')(x)
model = Model(inputs=old_model.input, outputs=output)

# 第一阶段：冻结主干，只训练输出层
for layer in old_model.layers:
    layer.trainable = False
model.layers[-1].trainable = True

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    r'D:\final\models_cicids2017_h5\EfficientNetB7_new.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)


# 第二阶段：解冻主干后50层，继续训练
for layer in old_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# 合并history用于画图
def merge_hist(hist1, hist2):
    h = dict(hist1.history)
    for k in hist2.history:
        if k in h:
            h[k] += hist2.history[k]
        else:
            h[k] = hist2.history[k]
    return h

full_history = merge_hist(history1, history2)

# 画图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(full_history['accuracy'], label='Train Accuracy')
plt.plot(full_history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(full_history['loss'], label='Train Loss')
plt.plot(full_history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('efficientnetb7_result.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# 测试
model.load_weights(r'D:\final\models_cicids2017_h5\EfficientNetB7_cicids2017.h5')
test_loss, test_acc = model.evaluate(test_generator)
print(f"测试集损失: {test_loss:.4f}, 准确度: {test_acc:.4f}")