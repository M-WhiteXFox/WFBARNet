# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image

# 获取工具目录路径
tool_dir = Path(__file__).resolve().parent
img_path = tool_dir / "hylogo.png"
ico_path = tool_dir / "app.ico"

# 保存为 ICO 文件
img = Image.open(img_path)
img.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(f"图标已保存到: {ico_path}")
