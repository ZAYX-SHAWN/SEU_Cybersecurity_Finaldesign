毕业设计：基于卷积神经网络的物联网设备入侵检测系统设计与实现
项目简介
本项目基于公开数据集，设计并实现了一个利用卷积神经网络（CNN）进行物联网设备入侵检测的系统。系统通过特征提取与预处理，将网络流量特征转化为适合 CNN 分析的二维图像格式，并基于多种主流预训练 CNN 架构进行模型迁移。同时，通过遗传算法优化模型超参数，并运用集成学习技术提升 IDS 性能。

系统实现流程
数据预处理与特征工程
merge.py
合并多个分散的数据集为一个数据集。
normalization.py
对数据集进行按列归一化处理。
spilit_dataset.py
按照标签将合并后的数据集分成 15 个不同类别。
transision.py
将预处理好的数据集转换为图像格式。
transision_to_224.py
将图像数据统一转换为 224×224 的尺寸。
模型训练与迁移学习
train_vgg16.py
利用预训练 VGG16 进行模型训练。
train_.py*
其他以 train_ 为前缀的脚本对应不同预训练模型的训练代码。
transfer_vgg16.py
迁移学习训练代码（以 CICIDS2017 数据集为例，与 train_* 用法相同）。
training.py
初期实验代码（目前已废弃，不推荐使用）。
模型调优与集成
HPO.py
利用遗传算法进行模型超参数优化（HyperParameter Optimization）。
aggregation.py
集成多个模型输出，提升最终检测性能。
结果分析与可视化
output_*.py
生成用于论文展示的各类图片和结果。
prediction.py
用测试集评估模型性能，输出混淆矩阵及测试结果。

目录结构：
.
├── aggregation.py
├── HPO.py
├── merge.py
├── normalization.py
├── output_1.py
├── output_2.py
├── output_3.py
├── prediction.py
├── spilit_dataset.py
├── train_vgg16.py
├── train_*.py
├── transfer_vgg16.py
├── transfer_*.py
├── transision.py
├── transision_to_224.py
└── training.py  # （已废弃）


使用说明
运行 merge.py 和 normalization.py 完成数据集的准备工作。
用 spilit_dataset.py 分类数据，transision.py 和 transision_to_224.py 将数据转为理念格式。
使用 train_*.py 或 transfer_*.py 脚本按需结合模型训练你的 CNN。
通过 HPO.py 进行超参数优化。
用 aggregation.py 集成多个模型的结果以增强检测能力。
最后通过 prediction.py 评估模型性能，并利用 output_.py 生成论文所需图表。
