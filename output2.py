from PIL import Image

# 读取两张图像
img1 = Image.open(r'D:\final\results_ciciot2023/EfficientNetV2L A.png')
img2 = Image.open(r'D:\final\results_ciciot2023/EfficientNetV2L L.png')

# 假设需要横向合并
# 计算合并后图像的宽高
width1, height1 = img1.size
width2, height2 = img2.size

# 统一高度（取两图较高的那个），可根据需求修改
max_height = max(height1, height2)
total_width = width1 + width2

# 创建新的空白图像, 白色背景
new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))

# 粘贴两张图像
new_img.paste(img1, (0, 0))
new_img.paste(img2, (width1, 0))

# 保存合并后的图像
new_img.save('merged_accuracy_loss.png')

print('合并完成，保存为 merged_accuracy_loss.png')