from typing import Optional
import pypdfium2 as pdfium
from PIL import Image


def pdf_bytes_to_image_first_page(pdf_bytes: bytes, dpi: int = 300) -> Image.Image:
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf.get_page(0)
    pil_image = page.render(scale=dpi / 72).to_pil()
    page.close()
    pdf.close()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image
