import os
import random
from PIL import Image, ImageDraw, ImageFont


def create_attack_type_collage(input_dir, output_path, grid_rows=3, grid_cols=5):
    # 参数设置
    tile_size = (224, 224)  # 单张图片尺寸
    padding = 10  # 图片间距
    font_size = 20  # 标签字体大小
    background_color = (255, 255, 255)  # 背景颜色

    # 获取攻击类型目录
    attack_types = [d for d in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, d))]
    attack_types.sort()

    selected_images = []

    # 随机选择并处理图片
    for attack_type in attack_types[:grid_rows * grid_cols]:  # 确保不超过15种
        attack_dir = os.path.join(input_dir, attack_type)

        # 获取所有图片文件
        img_files = [f for f in os.listdir(attack_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if img_files:
            # 随机选择一张
            selected_file = random.choice(img_files)
            img_path = os.path.join(attack_dir, selected_file)

            # 打开并处理图片
            with Image.open(img_path) as img:
                # 调整尺寸
                img = img.resize(tile_size, Image.BILINEAR)

                # 创建带标签的图片
                labeled_img = Image.new('RGB', (tile_size[0], tile_size[1] + 30), color=background_color)
                labeled_img.paste(img, (0, 0))

                # 添加文字标签
                draw = ImageDraw.Draw(labeled_img)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                text_width = draw.textlength(attack_type, font=font)
                draw.text(
                    ((tile_size[0] - text_width) / 2, tile_size[1] + 5),
                    attack_type,
                    fill=(0, 0, 0),
                    font=font
                )

                selected_images.append(labeled_img)

    # 创建拼接画布
    canvas_width = grid_cols * (tile_size[0] + padding) - padding
    canvas_height = grid_rows * (tile_size[1] + 30 + padding) - padding
    collage = Image.new('RGB', (canvas_width, canvas_height), color=background_color)

    # 排列图片
    for idx, img in enumerate(selected_images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (tile_size[0] + padding)
        y = row * (tile_size[1] + 30 + padding)
        collage.paste(img, (x, y))

    # 保存结果
    collage.save(output_path)
    print(f"拼接完成！结果保存至：{output_path}")


# 使用示例
if __name__ == "__main__":
    input_directory = r"D:\final\train_A_cicids2017"
    output_image = r"D:\final\attack_type_collage.png"
    create_attack_type_collage(input_directory, output_image)