# utils/ocr.py

from PIL import Image
import pytesseract

def image_to_text(image_path, lang="eng"):
    """
    使用 pytesseract 将图像转换为文本。

    Args:
        image_path (str): 图像文件路径
        lang (str): 语言（默认英文）

    Returns:
        str: 提取出的纯文本
    """
    image = Image.open(image_path)  # 用 PIL 打开图像
    text = pytesseract.image_to_string(image, lang=lang)  # 调用 tesseract 进行 OCR 识别
    return text.strip()  # 去掉开头结尾空格和换行
