from PIL import Image
import pytesseract

def image_to_text(image_path, lang="eng"):
    """
    使用 pytesseract 将图像转换为文本。
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text.strip()
