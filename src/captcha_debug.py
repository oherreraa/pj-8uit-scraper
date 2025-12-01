#!/usr/bin/env python3
"""
captcha_debug.py

Script independiente para probar solo la resolución del CAPTCHA
de la página:

https://sap.pj.gob.pe/portalabastecimiento-web/Convocatorias8uit

Flujo:
- Abre la página.
- Ubica la imagen del CAPTCHA mediante DOM.
- Recorta y guarda la imagen (raw).
- Ejecuta varios preprocesados + Tesseract.
- Muestra todos los intentos y el mejor valor detectado.
"""

import asyncio
import os
import io
from datetime import datetime
from typing import List, Tuple

from playwright.async_api import async_playwright, Page
from PIL import Image, ImageOps, ImageFilter
import pytesseract


URL = "https://sap.pj.gob.pe/portalabastecimiento-web/Convocatorias8uit"


# ---------------------------------------------------------------------------
# Utilidades de imagen / OCR
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Limpia el texto devuelto por Tesseract."""
    return "".join(c for c in text if c.isalnum()).upper()


def ocr_attempts(image: Image.Image) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Ejecuta múltiples intentos de OCR sobre la misma imagen,
    devolviendo:
      - mejor_texto
      - lista de (descripcion, texto_limpio)
    """
    attempts: List[Tuple[str, str]] = []

    # Distintas configuraciones Tesseract
    configs = [
        ("psm8_whitelist", "--oem 3 --psm 8 -l eng "
                           "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("psm7_whitelist", "--oem 3 --psm 7 -l eng "
                           "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("psm6_whitelist", "--oem 3 --psm 6 -l eng "
                           "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("psm8_no_whitelist", "--oem 3 --psm 8 -l eng"),
    ]

    # Distintos preprocesados
    variants = []

    base_gray = image.convert("L")
    variants.append(("gray", base_gray))

    # Escalados
    for scale in (2, 3, 4):
        w, h = base_gray.size
        resized = base_gray.resize((w * scale, h * scale))
        variants.append((f"gray_x{scale}", resized))

        # Binarizaciones simples
        for thr in (120, 140, 160, 180):
            bin_im = resized.point(lambda x, t=thr: 0 if x < t else 255, "1")
            variants.append((f"bin_x{scale}_thr{thr}", bin_im))

        # Invertido + filtro
        inv = ImageOps.invert(resized)
        inv = inv.filter(ImageFilter.MedianFilter(size=3))
        variants.append((f"invert_med_x{scale}", inv))

    # Ejecutar todas las combinaciones
    for v_name, img_v in variants:
        for cfg_name, cfg in configs:
            desc = f"{v_name} | {cfg_name}"
            try:
                raw_text = pytesseract.image_to_string(img_v, config=cfg)
                clean = _clean_text(raw_text)
                attempts.append((desc, clean))
            except Exception as e:
                attempts.append((desc, f"ERROR:{e}"))

    # Elegir mejor por heurística muy sencilla:
    # - Preferir longitud 4 (el captcha suele ser 4 chars).
    # - Si hay varios, el primero no vacío.
    best = ""
    for desc, txt in attempts:
        if 4 <= len(txt) <= 6:
            best = txt
            break
    if not best:
        # Si no encontramos longitud ideal, tomar el texto no vacío más largo
        candidates = [txt for _, txt in attempts if txt and not txt.startswith("ERROR:")]
        if candidates:
            best = max(candidates, key=len)
        else:
            best = ""

    return best, attempts


# ---------------------------------------------------------------------------
# Playwright: captura del CAPTCHA
# ---------------------------------------------------------------------------

async def locate_captcha_img(page: Page):
    selectors = [
        'img[src*="captcha"]',
        'img[alt*="captcha"]',
        'img[id*="captcha"]',
        ".captcha img",
        "#captcha img",
        'img[src*="Codigo"]',
        'img[src*="codigo"]',
    ]
    for sel in selectors:
        try:
            img = page.locator(sel).first
            if await img.is_visible(timeout=3000):
                print(f"🤖 CAPTCHA encontrado con selector: {sel}")
                return img
        except Exception:
            continue
    print("❌ No se encontró imagen de CAPTCHA.")
    return None


async def capture_captcha_image(page: Page) -> Image.Image | None:
    img = await locate_captcha_img(page)
    if img is None:
        return None

    bbox = await img.bounding_box()
    if not bbox:
        print("❌ No se pudo obtener bounding box del CAPTCHA.")
        return None

    screenshot_bytes = await page.screenshot()
    full = Image.open(io.BytesIO(screenshot_bytes))

    padding = 4
    left = max(0, int(bbox["x"] - padding))
    top = max(0, int(bbox["y"] - padding))
    right = min(full.width, int(bbox["x"] + bbox["width"] + padding))
    bottom = min(full.height, int(bbox["y"] + bbox["height"] + padding))

    captcha = full.crop((left, top, right, bottom))

    os.makedirs("captcha_images", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"captcha_images/debug_captcha_raw_{ts}.png"
    captcha.save(raw_path)
    print(f"💾 CAPTCHA guardado en: {raw_path}")

    return captcha


# ---------------------------------------------------------------------------
# Main asincrónico
# ---------------------------------------------------------------------------

async def main(headless: bool = True) -> None:
    print("🚀 Iniciando captura de CAPTCHA para debug...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.goto(URL, timeout=60000)
        await page.wait_for_load_state("networkidle", timeout=60000)

        captcha_image = await capture_captcha_image(page)
        await browser.close()

    if captcha_image is None:
        print("❌ No hay imagen de CAPTCHA para procesar.")
        return

    print("👁️ Lanzando intentos de OCR...")
    best, attempts = ocr_attempts(captcha_image)

    print("\n===== RESUMEN INTENTOS OCR =====")
    for desc, txt in attempts:
        if txt.startswith("ERROR:"):
            print(f"[{desc}] -> {txt}")
        else:
            print(f"[{desc}] -> '{txt}' (len={len(txt)})")

    print("\n===== MEJOR RESULTADO =====")
    if best:
        print(f"✅ Mejor texto detectado: '{best}'")
    else:
        print("❌ No se pudo obtener ningún texto útil del CAPTCHA.")


if __name__ == "__main__":
    # Para depuración local puedes cambiar headless=False
    asyncio.run(main(headless=True))
