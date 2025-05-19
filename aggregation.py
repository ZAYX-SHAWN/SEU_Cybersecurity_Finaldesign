import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import scikitplot as skplt
import matplotlib.pyplot as plt

# 模型在如下路径
model_paths = [
    'D:/final\models_hpo/CNN_cicids2017.h5',
    'D:/final/models_hpo/EfficientNetB7_newhpo.h5',
    'D:/final\models_hpo/EfficientNetV2L_newhpo.h5',
    'D:/final\models_hpo/VGG16_new.h5',
    'D:/final\models_hpo/VGG19_new.h5',
]

# 加载模型
models = [load_model(p) for p in model_paths]

# 加载测试数据
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    r'D:\final\test_A_cicids2017',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 记录所有模型的预测，注意预测要与test数据顺序一一对应
all_preds = []
for model in models:
    preds = model.predict(test_generator, verbose=1)
    all_preds.append(preds)

# 将所有概率预测堆叠，计算平均
ensemble_proba = np.mean(all_preds, axis=0)  # shape仍为[N, num_classes]
ensemble_pred_labels = np.argmax(ensemble_proba, axis=1)

# 获得真实标签
real_labels = test_generator.classes

# 类别名映射
class_indices = test_generator.class_indices
inv_class_indices = {v: k for k, v in class_indices.items()}
ensemble_pred_names = [inv_class_indices[i] for i in ensemble_pred_labels]
real_label_names = [inv_class_indices[i] for i in real_labels]

# 评估ensemble效果
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Ensemble accuracy:", accuracy_score(real_label_names, ensemble_pred_names))
print(classification_report(real_label_names, ensemble_pred_names, digits=4))

# (可选) 展示混淆矩阵

skplt.metrics.plot_confusion_matrix(real_label_names, ensemble_pred_names, normalize=True, x_tick_rotation=90, figsize=(12,12), cmap='Blues')
plt.title('Ensemble Confusion Matrix')
plt.show()