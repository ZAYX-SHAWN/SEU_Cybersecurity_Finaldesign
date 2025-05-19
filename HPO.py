import numpy as np
import pygad
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -- 固定参数
train_dir = r'D:\final\train_A_cicids2017'
img_size = (224, 224)
NUM_CLASSES = 15  # 根据你实际情况修改

# -- 搜索空间
FIRST_LRS = [1e-3, 5e-4, 1e-4]
SECOND_LRS = [1e-4, 5e-5, 1e-5]
UNFREEZE_LAYERS = [20, 50, 100]
BATCH_SIZES = [16, 32]
DROPOUTS = [0.0, 0.3, 0.5]  # 这里只做示例，后面可扩展用


# 遗传算法解码器
def decode_solution(solution):
    first_lr = FIRST_LRS[int(solution[0])]
    second_lr = SECOND_LRS[int(solution[1])]
    unfreeze_num = UNFREEZE_LAYERS[int(solution[2])]
    batch_size = BATCH_SIZES[int(solution[3])]
    dropout = DROPOUTS[int(solution[4])]
    return first_lr, second_lr, unfreeze_num, batch_size, dropout


# 适应度函数
def fitness_func(ga_instance, solution, solution_idx):
    first_lr, second_lr, unfreeze_num, batch_size, dropout = decode_solution(solution)
    # -- 数据加载（每次用mini数据满足加速实验，也可固定随机种子）
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical',
        subset='training', shuffle=True, seed=42)
    val_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical',
        subset='validation', shuffle=True, seed=42)
    NUM_CLASSES = train_generator.num_classes

    # -- 模型加载/搭建
    old_model = load_model(r'D:\final\models_ciciot2023_h5/efficientnetb7.h5')
    x = old_model.layers[-2].output
    x = Dropout(dropout, name=f"ga_dropout_{solution_idx}")(x)
    output = Dense(NUM_CLASSES, activation='softmax', name=f"ga_dense_out_{solution_idx}")(x)
    model = Model(inputs=old_model.input, outputs=output)

    # -- 第一阶段
    for layer in old_model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True
    model.compile(optimizer=Adam(first_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=0)
    history1 = model.fit(
        train_generator,
        epochs=5,  # GA 训练周期建议缩短，加快搜索
        validation_data=val_generator,
        callbacks=[early_stop], verbose=1
    )
    # -- 第二阶段
    for layer in old_model.layers[-unfreeze_num:]:
        layer.trainable = True
    model.compile(optimizer=Adam(second_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    history2 = model.fit(
        train_generator,
        epochs=5,  # 缩短, 加快
        validation_data=val_generator,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    # -- 返回指标
    val_acc = max(history2.history['val_accuracy'])
    return val_acc


# -- 遗传算法参数
gene_space = [
    range(len(FIRST_LRS)),
    range(len(SECOND_LRS)),
    range(len(UNFREEZE_LAYERS)),
    range(len(BATCH_SIZES)),
    range(len(DROPOUTS))
]

ga_instance = pygad.GA(
    num_generations=4,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=8,
    num_genes=len(gene_space),
    gene_space=gene_space,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20,
)

ga_instance.run()

# -- 输出结果
solution, solution_fitness, _ = ga_instance.best_solution()
first_lr, second_lr, unfreeze_num, batch_size, dropout = decode_solution(solution)
print("Best Params:", first_lr, second_lr, unfreeze_num, batch_size, dropout)
print("Best Fitness (val_acc):", solution_fitness)

ga_instance.plot_fitness()