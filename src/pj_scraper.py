#!/usr/bin/env python3
"""
PJ 8UIT Scraper

Objetivo:
- Abrir https://sap.pj.gob.pe/portalabastecimiento-web/Convocatorias8uit
- Detectar CAPTCHA vía DOM.
- Capturar la imagen del CAPTCHA y resolverla con Tesseract (OCR avanzado),
  o usar un código fijo si viene por variable de entorno.
- Escribir el CAPTCHA en el input.
- Hacer clic en "Buscar" (con fallback via DOM).
- Verificar si:
    * el CAPTCHA fue incorrecto (mensaje en la página), o
    * se cargó la tabla con filas de convocatorias.
- Si el CAPTCHA falló, reintentar varias veces (recargando página y nuevo CAPTCHA).
- Cuando se obtenga una tabla con filas válidas:
    * Extraer todas las filas de la página.
    * Recorrer páginas con "Siguiente".
    * Para cada fila:
        - Intentar localizar el elemento que descarga el TDR.
        - Usar page.expect_download() para capturar el PDF.
        - Extraer el bloque "CARACTERISTICAS TECNICAS" del PDF.
        - Si no se encuentra con texto embebido:
            -> activar OCR sobre el PDF y volver a buscar el capítulo.
    * Emitir UN SOLO JSON "procesado" para GitHub Actions (data/processed/*.json).
    * En cada fila dejar:
        - numero_convocatoria
        - unidad_organica
        - descripcion
        - cierre_postulacion (texto original)
        - cierre_postulacion_lima (ISO 8601 con tz America/Lima)
        - tdr_filename (nombre del PDF descargado, si se logró)
        - caracteristicas_tecnicas (bloque extraído del PDF, si se logró)
        - caracteristicas_tecnicas_ocr (bool: True si se usó OCR)
"""

import asyncio
import json
import os
import sys
import io
import re
import subprocess
import glob
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo
from playwright.async_api import async_playwright, Page

from PIL import Image, ImageOps, ImageFilter
import pytesseract
from PyPDF2 import PdfReader


URL = "https://sap.pj.gob.pe/portalabastecimiento-web/Convocatorias8uit"
LIMA_TZ = ZoneInfo("America/Lima")


# ---------------------------------------------------------------------------
# Utilidades OCR / PDF / debug
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Limpia texto devuelto por Tesseract (solo alfanumérico y mayúsculas)."""
    return "".join(c for c in text if c.isalnum()).upper()


def solve_captcha_tesseract_advanced(image: Image.Image) -> Optional[str]:
    """
    Versión avanzada de OCR:
    - Varios preprocesados.
    - Varias configuraciones Tesseract.
    - Devuelve el mejor candidato (4–6 caracteres preferido).
    """
    configs = [
        (
            "psm8_whitelist",
            "--oem 3 --psm 8 -l eng "
            "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ),
        (
            "psm7_whitelist",
            "--oem 3 --psm 7 -l eng "
            "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ),
        (
            "psm6_whitelist",
            "--oem 3 --psm 6 -l eng "
            "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ),
        ("psm8_no_whitelist", "--oem 3 --psm 8 -l eng"),
    ]

    variants: List[tuple[str, Image.Image]] = []

    base_gray = image.convert("L")
    variants.append(("gray", base_gray))

    for scale in (2, 3, 4):
        w, h = base_gray.size
        resized = base_gray.resize((w * scale, h * scale))
        variants.append((f"gray_x{scale}", resized))

        for thr in (120, 140, 160, 180):
            bin_im = resized.point(lambda x, t=thr: 0 if x < t else 255, "1")
            variants.append((f"bin_x{scale}_thr{thr}", bin_im))

        inv = ImageOps.invert(resized)
        inv = inv.filter(ImageFilter.MedianFilter(size=3))
        variants.append((f"invert_med_x{scale}", inv))

    attempts: List[tuple[str, str]] = []

    for v_name, img_v in variants:
        for cfg_name, cfg in configs:
            desc = f"{v_name} | {cfg_name}"
            try:
                raw = pytesseract.image_to_string(img_v, config=cfg)
                clean = _clean_text(raw)
                attempts.append((desc, clean))
            except Exception as e:
                attempts.append((desc, f"ERROR:{e}"))

    print("📜 Intentos de OCR:")
    for desc, txt in attempts:
        if txt.startswith("ERROR:"):
            print(f"  [{desc}] -> {txt}")
        else:
            print(f"  [{desc}] -> '{txt}' (len={len(txt)})")

    best = ""
    # Preferimos cadenas de longitud 4–6 (tamaño típico de captcha)
    for desc, txt in attempts:
        if 4 <= len(txt) <= 6 and not txt.startswith("ERROR:"):
            best = txt
            break

    if not best:
        candidates = [
            txt for _, txt in attempts if txt and not txt.startswith("ERROR:")
        ]
        if candidates:
            best = max(candidates, key=len)

    if best:
        print(f"✅ Mejor resultado OCR: '{best}'")
        return best

    print("❌ No se obtuvo un resultado OCR usable.")
    return None


def ocr_pdf_to_text(pdf_path: str) -> str:
    """
    Convierte el PDF completo a imágenes (pdftoppm) y les aplica OCR (pytesseract).
    Retorna texto concatenado de todas las páginas OCR.
    """
    texts: List[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "page")
            # Requiere poppler-utils (pdftoppm)
            subprocess.run(
                ["pdftoppm", "-png", pdf_path, prefix],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            png_files = sorted(glob.glob(os.path.join(tmpdir, "page-*.png")))
            for png in png_files:
                try:
                    img = Image.open(png).convert("L")
                    img = img.filter(ImageFilter.MedianFilter(size=3))
                    txt = pytesseract.image_to_string(img, lang="spa+eng")
                    if txt.strip():
                        texts.append(txt)
                except Exception as e:
                    print(f"⚠️ Error OCR en página {png}: {e}")
    except Exception as e:
        print(f"⚠️ Error en pipeline pdftoppm/OCR: {e}")

    return "\n".join(texts).strip()


def _extract_caracteristicas_from_text(full_text: str) -> Optional[str]:
    """
    Busca el capítulo de CARACTERÍSTICAS / ESPECIFICACIONES TÉCNICAS
    en un texto plano de todo el PDF. Devuelve el bloque o None.
    """
    if not full_text:
        return None

    norm = full_text.upper()

    # Encabezados típicos
    patterns = [
        r"CARACTER[IÍ]STICAS\s+T[ÉE]CNICAS",
        r"ESPECIFICACIONES\s+T[ÉE]CNICAS",
    ]

    start_idx = -1
    for pat in patterns:
        m = re.search(pat, norm)
        if m:
            start_idx = m.start()
            break

    if start_idx == -1:
        return None

    # Posibles marcadores de fin de sección
    end_markers = [
        "CONDICIONES GENERALES",
        "CONDICIONES CONTRACTUALES",
        "CONDICIONES",
        "REQUISITOS",
        "OBLIGACIONES",
        "PLAZO DE ENTREGA",
        "PLAZO DE EJECUCION",
        "PLAZO DE EJECUCIÓN",
        "GARANTIAS",
        "GARANTÍAS",
        "FORMA DE PAGO",
    ]

    end_idx = len(full_text)
    for m in end_markers:
        pos = norm.find(m, start_idx + 10)
        if pos != -1 and pos > start_idx:
            end_idx = min(end_idx, pos)

    segment = full_text[start_idx:end_idx].strip()
    if not segment:
        return None

    max_len = 4000
    if len(segment) > max_len:
        segment = segment[:max_len] + "\n[...]"

    return segment


def extract_caracteristicas_from_pdf(pdf_path: str) -> Tuple[Optional[str], bool]:
    """
    Extrae el capítulo de CARACTERISTICAS TECNICAS con 2 etapas:

    1) Lectura por texto embebido (PyPDF2).
       - Si encuentra el capítulo → devuelve (texto, False).

    2) Si no se encontró, activa OCR (pdftoppm + Tesseract).
       - Si encuentra el capítulo → devuelve (texto, True).
       - Si no → (None, True) para indicar que se intentó OCR.
    """
    # --- Etapa 1: PyPDF2 (texto embebido) ---
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"⚠️ No se pudo abrir PDF '{pdf_path}': {e}")
        return None, False

    pages_text: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pages_text.append(t)

    full_text = "\n".join(pages_text)
    # Buscar capítulo en texto embebido
    segment_plain = _extract_caracteristicas_from_text(full_text)
    if segment_plain:
        return segment_plain, False

    # --- Etapa 2: OCR completo del PDF ---
    print(f"ℹ️ No se encontró capítulo 'CARACTERISTICAS TECNICAS' con texto embebido. Activando OCR para {pdf_path}...")
    ocr_text = ocr_pdf_to_text(pdf_path)
    if not ocr_text:
        print("⚠️ OCR no devolvió texto util.")
        return None, True

    segment_ocr = _extract_caracteristicas_from_text(ocr_text)
    if segment_ocr:
        return segment_ocr, True

    print("⚠️ OCR ejecutado, pero no se encontró capítulo específico de características técnicas.")
    return None, True


async def debug_dump_page(page: Page, label: str = "after_search") -> None:
    """Guardado de screenshot + HTML de la página para debug."""
    try:
        os.makedirs("data/raw", exist_ok=True)
        ts = datetime.now(LIMA_TZ).strftime("%Y%m%d_%H%M%S")
        png_path = f"data/raw/debug_{label}_{ts}.png"
        html_path = f"data/raw/debug_{label}_{ts}.html"

        await page.screenshot(path=png_path, full_page=True)
        html = await page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        print("🧪 Debug dump guardado:")
        print(f"   PNG:  {png_path}")
        print(f"   HTML: {html_path}")
    except Exception as e:
        print(f"⚠️ Error en debug_dump_page: {e}")


# ---------------------------------------------------------------------------
# Clase principal del scraper
# ---------------------------------------------------------------------------

class PJScraper:
    def __init__(
        self,
        headless: bool = True,
        timeout: int = 60,
        captcha_code: Optional[str] = None,
        max_pages: int = 30,
        max_captcha_attempts: int = 5,
        use_tesseract: bool = True,
    ) -> None:
        self.base_url = "https://sap.pj.gob.pe/portalabastecimiento-web"
        self.url = f"{self.base_url}/Convocatorias8uit"
        self.headless = headless
        self.timeout_ms = timeout * 1000
        self.captcha_code = captcha_code
        self.max_pages = max_pages
        self.max_captcha_attempts = max_captcha_attempts
        self.use_tesseract = use_tesseract

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
        }

    # ----------------------------- Setup -----------------------------------

    async def setup_page(self, page: Page) -> None:
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.set_extra_http_headers(self.headers)

        page.on("popup", lambda popup: asyncio.create_task(popup.close()))
        page.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))

        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            """
        )

    async def handle_overlays(self, page: Page) -> None:
        """Cerrar modales / banners de cookies si aparecen."""
        try:
            modal_selectors = [
                ".modal:visible",
                ".popup:visible",
                ".dialog:visible",
                '[role="dialog"]:visible',
            ]
            for sel in modal_selectors:
                try:
                    loc = page.locator(sel).first
                    if await loc.is_visible(timeout=2000):
                        await page.keyboard.press("Escape")
                        print(f"✅ Modal cerrado: {sel}")
                        break
                except Exception:
                    continue

            cookie_selectors = [
                'button:has-text("Aceptar")',
                'button:has-text("Accept")',
                ".cookie-accept",
                ".accept-cookies",
            ]
            for sel in cookie_selectors:
                try:
                    btn = page.locator(sel).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        print("🍪 Cookies aceptadas")
                        break
                except Exception:
                    continue

        except Exception as e:
            print(f"⚠️ Error manejando overlays: {e}")

    # --------------------------- CAPTCHA -----------------------------------

    async def locate_captcha_img(self, page: Page):
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
        print("ℹ️ No se encontró imagen de CAPTCHA.")
        return None

    async def has_captcha(self, page: Page) -> bool:
        img = await self.locate_captcha_img(page)
        return img is not None

    async def capture_captcha_image(self, page: Page) -> Optional[Image.Image]:
        img = await self.locate_captcha_img(page)
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
        ts = datetime.now(LIMA_TZ).strftime("%Y%m%d_%H%M%S")
        raw_path = f"captcha_images/captcha_raw_{ts}.png"
        captcha.save(raw_path)
        print(f"💾 CAPTCHA capturado en: {raw_path}")

        return captcha

    async def fill_captcha_and_click_search_once(self, page: Page, attempt: int) -> None:
        """
        Resolver CAPTCHA (si existe), rellenar input y pulsar Buscar.
        SOLO UN INTENTO; el bucle de reintentos está en solve_and_search_with_retries().
        """
        print(f"🎯 Intento de búsqueda con CAPTCHA #{attempt}")

        if not await self.has_captcha(page):
            print("ℹ️ No se detectó CAPTCHA, continuando sin resolverlo.")
        else:
            captcha_text: Optional[str] = None

            if self.captcha_code:
                captcha_text = self.captcha_code.strip()
                print(f"🔑 Usando CAPTCHA fijo: '{captcha_text}'")
            elif self.use_tesseract:
                img = await self.capture_captcha_image(page)
                if img is not None:
                    captcha_text = solve_captcha_tesseract_advanced(img)

            if not captcha_text:
                print("⚠️ No se obtuvo texto de CAPTCHA. La búsqueda probablemente falle.")

            input_loc = None
            input_selectors = [
                'input[name*="captcha"]',
                'input[id*="captcha"]',
                'input[placeholder*="aptcha"]',
                "input[type='text']",
            ]
            for sel in input_selectors:
                try:
                    cand = page.locator(sel).first
                    if await cand.is_visible(timeout=2000):
                        input_loc = cand
                        print(f"📝 Input de CAPTCHA: {sel}")
                        break
                except Exception:
                    continue

            if input_loc is not None and captcha_text:
                await input_loc.fill("")
                await input_loc.type(captcha_text, delay=80)
                print("✅ CAPTCHA escrito en el input.")
            elif input_loc is None:
                print("⚠️ No se encontró input de CAPTCHA.")

        # Botón Buscar
        search_selectors = [
            'button:has-text("Buscar")',
            'input[value*="Buscar"]',
            'input[type="submit"]',
            'button[type="submit"]',
        ]
        clicked = False
        for sel in search_selectors:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=3000):
                    await btn.scroll_into_view_if_needed()
                    await btn.click()
                    print(f"🔎 Click en botón Buscar (locator): {sel}")
                    clicked = True
                    break
            except Exception:
           


::contentReference[oaicite:0]{index=0}
