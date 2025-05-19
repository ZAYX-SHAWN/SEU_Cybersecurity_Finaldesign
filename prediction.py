import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Generate Images from Test Set
TARGET_SIZE = (224, 224)
INPUT_SIZE = (224, 224, 3)
BATCHSIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)
generator = test_datagen.flow_from_directory(
    r'D:\final\test_A_cicids2017',
    target_size=TARGET_SIZE,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# load model
model = load_model('D:/final/models_hpo/CNN_cicids2017.h5')

# 获取所有图片的预测结果
preds = model.predict(generator, steps=generator.samples)
pred_labels = np.argmax(preds, axis=1)

# 类别索引反转
inv_class_indices = {v: k for k, v in generator.class_indices.items()}
pred_label_names = [inv_class_indices[i] for i in pred_labels]

# 获取真实标签
real_labels = generator.classes               # int型label
real_label_names = [inv_class_indices[i] for i in real_labels]  # 类别名

# 输出主要指标
acc = accuracy_score(real_label_names, pred_label_names)
precision = precision_score(real_label_names, pred_label_names, average='weighted')
recall = recall_score(real_label_names, pred_label_names, average='weighted')
f1 = f1_score(real_label_names, pred_label_names, average='weighted')

print("模型准确率(Accuracy): {:.4f}".format(acc))
print("模型精确率(Precision): {:.4f}".format(precision))
print("模型召回率(Recall): {:.4f}".format(recall))
print("模型F1-score: {:.4f}".format(f1))

# 详细分类报告
print("\n详细分类报告：")
print("Overall Accuracy: {:.4f}\n".format(acc))
print(classification_report(real_label_names, pred_label_names, digits=4))
# 可视化混淆矩阵
skplt.metrics.plot_confusion_matrix(real_label_names, pred_label_names,
                                    normalize=True,
                                    x_tick_rotation=90,
                                    figsize=(10, 10),
                                    cmap='Blues')
plt.title('CNN Matrix', fontsize=16, fontweight='bold')
plt.show()