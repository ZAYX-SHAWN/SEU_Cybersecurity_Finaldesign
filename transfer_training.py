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

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 1. 加载已训练模型
base_model = load_model('CNN.h5')

# 2. 替换分类头
x = base_model.layers[-2].output  # -2是Dropout层
new_output = Dense(new_num_class, activation='softmax')(x)
transfer_model = Model(inputs=base_model.input, outputs=new_output)

# 3. 冻结前面所有层
for layer in transfer_model.layers[:-2]:  # 保证只训练新Dense层和Dropout层
    layer.trainable = False

# 4. 编译
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
transfer_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 5. 重新训练
history = transfer_model.fit(
    new_train_generator,
    steps_per_epoch=len(new_train_generator),
    epochs=new_epochs,
    validation_data=new_validation_generator,
    validation_steps=len(new_validation_generator),
    callbacks=[saveBestModel]
)

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