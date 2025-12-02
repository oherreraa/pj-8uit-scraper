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
        - Extraer el bloque "CARACTERISTICAS TECNICAS" del PDF, con OCR si es imagen.
        - Reorganizar esa información en un solo string con varios párrafos
          (en orden de Ítem) y almacenarla en `caracteristicas_tecnicas`.
    * Guardar resultados en JSON (raw + procesado) para GitHub Actions.
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
    Versión avanzada de OCR para CAPTCHA:
    - Varios preprocesados.
    - Varias configuraciones Tesseract.
    - Devuelve el mejor candidato (4–6 caracteres preferido).
    """
    configs: List[Tuple[str, str]] = [
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

    variants: List[Tuple[str, Image.Image]] = []

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

    attempts: List[Tuple[str, str]] = []

    for v_name, img_v in variants:
        for cfg_name, cfg in configs:
            desc = f"{v_name} | {cfg_name}"
            try:
                raw = pytesseract.image_to_string(img_v, config=cfg)
                clean = _clean_text(raw)
                attempts.append((desc, clean))
            except Exception as e:
                attempts.append((desc, f"ERROR:{e}"))

    print("📜 Intentos de OCR (CAPTCHA):")
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
        print(f"✅ Mejor resultado OCR CAPTCHA: '{best}'")
        return best

    print("❌ No se obtuvo un resultado OCR usable para CAPTCHA.")
    return None


def extract_text_from_pdf_with_ocr(pdf_path: str) -> str:
    """
    Intenta extraer texto de un PDF.
    1) Primero usa PyPDF2 (texto embebido).
    2) Si el resultado es muy pobre (típico de PDFs escaneados),
       hace fallback a OCR:
       - Convierte el PDF a imágenes PNG con 'pdftoppm' (poppler-utils).
       - Aplica Tesseract sobre cada página (lang='spa+eng').
    Devuelve un string con todo el texto concatenado.
    """
    text_parts: List[str] = []

    # --- Paso 1: PyPDF2 (texto embebido) ---
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    except Exception as e:
        print(f"⚠️ Error leyendo PDF con PyPDF2 '{pdf_path}': {e}")

    text_py = "\n".join(text_parts).strip()

    # Si el texto embebido parece razonable, lo usamos
    if len(text_py) > 500:  # umbral heurístico
        print(f("📄 Texto PyPDF2 usado (len={len(text_py)}) para {pdf_path}"))
        return text_py

    print(f"ℹ️ Texto embebido insuficiente (len={len(text_py)}). Fallback a OCR: {pdf_path}")

    # --- Paso 2: OCR sobre imágenes ---
    ocr_text_parts: List[str] = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "page")
            # Genera page-1.png, page-2.png, etc.
            cmd = ["pdftoppm", "-png", pdf_path, out_prefix]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            png_files = sorted(glob.glob(os.path.join(tmpdir, "page-*.png")))
            if not png_files:
                print(f"⚠️ pdftoppm no generó imágenes para {pdf_path}")
            else:
                print(f"🖼️ OCR sobre {len(png_files)} páginas para {pdf_path}")

            for png_path in png_files:
                try:
                    img = Image.open(png_path)
                    # ligero preprocesado (gris + filtro mediana)
                    img = img.convert("L")
                    img = img.filter(ImageFilter.MedianFilter(size=3))
                    # OCR en español+inglés
                    txt = pytesseract.image_to_string(img, lang="spa+eng")
                    if txt.strip():
                        ocr_text_parts.append(txt)
                except Exception as e:
                    print(f"⚠️ Error OCR en {png_path}: {e}")

    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error ejecutando pdftoppm para {pdf_path}: {e}")
    except Exception as e:
        print(f"⚠️ Error general en OCR para {pdf_path}: {e}")

    text_ocr = "\n".join(ocr_text_parts).strip()

    # Prioridad: si OCR dio algo, lo usamos; si no, devolvemos lo poco que hubo de PyPDF2
    if text_ocr:
        print(f"📄 Texto OCR usado (len={len(text_ocr)}) para {pdf_path}")
        return text_ocr

    print(f"⚠️ No se obtuvo texto útil ni con PyPDF2 ni con OCR para {pdf_path}")
    return text_py


def parse_caracteristicas_table(segment: str) -> Optional[List[Dict[str, str]]]:
    """
    Intenta interpretar el bloque de 'CARACTERISTICAS TECNICAS' como una tabla.
    Heurística:
      - Busca una línea de encabezado con palabras tipo ITEM / DESCRIPCIÓN / UND / CANTIDAD.
      - Usa separadores por "dos o más espacios" para dividir columnas.
      - Cada línea siguiente se mapea a columnas.

    Devuelve una lista de dicts {columna: valor} o None si no logra interpretar.
    """
    lines = [ln.strip() for ln in segment.splitlines()]
    lines = [ln for ln in lines if ln]  # quitar vacías

    if not lines:
        return None

    header_idx = -1
    header_cols: List[str] = []

    # Buscamos encabezado
    for i, ln in enumerate(lines):
        upper = ln.upper()
        if ("ITEM" in upper or "ÍTEM" in upper) and (
            "DESCRIPCION" in upper or "DESCRIPCIÓN" in upper
        ):
            header_idx = i
            header_cols = re.split(r"\s{2,}", ln)
            header_cols = [c.strip() for c in header_cols if c.strip()]
            break

    if header_idx == -1 or not header_cols:
        # No encontramos encabezado claro
        return None

    rows: List[Dict[str, str]] = []
    # Procesamos desde la línea siguiente al encabezado
    for ln in lines[header_idx + 1 :]:
        upper = ln.upper()
        # Cortar si parece que ya pasamos a otra sección
        if any(
            kw in upper
            for kw in [
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
        ):
            break

        # Dividir columnas por bloques de 2+ espacios
        parts = re.split(r"\s{2,}", ln)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue

        # Si hay menos partes que columnas, rellenamos con vacío
        if len(parts) < len(header_cols):
            parts = parts + [""] * (len(header_cols) - len(parts))
        # Si hay más partes, pegamos las extras en la última columna
        if len(parts) > len(header_cols):
            parts = parts[: len(header_cols) - 1] + [" ".join(parts[len(header_cols) - 1 :])]

        row = {header_cols[i]: parts[i] for i in range(len(header_cols))}
        rows.append(row)

    return rows or None


def extract_caracteristicas_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extrae del PDF el bloque 'CARACTERISTICAS TECNICAS' y lo reorganiza
    en un solo string con diferentes párrafos, ordenado por Ítem cuando
    sea posible. Si no se puede interpretar la tabla, se devuelve el
    segmento crudo recortado.
    """
    full_text = extract_text_from_pdf_with_ocr(pdf_path)
    if not full_text:
        return None

    normalized = full_text.upper()

    patterns = [
        "CARACTERISTICAS TECNICAS",
        "CARACTERÍSTICAS TÉCNICAS",
        "CARACTERISTICAS TÉCNICAS",
        "CARACTERÍSTICAS TECNICAS",
    ]

    start_idx = -1
    chosen = ""
    for p in patterns:
        pos = normalized.find(p)
        if pos != -1:
            start_idx = pos
            chosen = p
            break

    if start_idx == -1:
        print(f"ℹ️ No se encontró encabezado 'CARACTERISTICAS TECNICAS' en {pdf_path}")
        return None

    # Detectar fin de la sección por palabras clave típicas
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
        pos = normalized.find(m, start_idx + len(chosen))
        if pos != -1 and pos > start_idx:
            end_idx = min(end_idx, pos)

    segment = full_text[start_idx:end_idx].strip()

    # Intentar interpretarlo como tabla
    tabla = parse_caracteristicas_table(segment)
    if tabla:
        headers = list(tabla[0].keys())

        def find_key(predicate) -> Optional[str]:
            for k in headers:
                if predicate(k.upper()):
                    return k
            return None

        item_key = find_key(lambda u: "ITEM" in u or "ÍTEM" in u)
        desc_key = find_key(lambda u: "DESCRIPC" in u)
        und_key = find_key(lambda u: "UND" in u or "U.M" in u or "UNIDAD" in u)
        cant_key = find_key(lambda u: "CANT" in u)
        extra_keys = [
            k for k in headers
            if k not in {item_key, desc_key, und_key, cant_key} and k
        ]

        def sort_key(row: Dict[str, str]) -> int:
            if not item_key:
                return 0
            val = (row.get(item_key) or "").strip()
            m = re.search(r"\d+", val)
            if not m:
                return 0
            try:
                return int(m.group())
            except Exception:
                return 0

        ordered_rows = sorted(tabla, key=sort_key)

        paragraphs: List[str] = []
        for row in ordered_rows:
            item_val = (row.get(item_key) or "").strip() if item_key else ""
            desc_val = (row.get(desc_key) or "").strip() if desc_key else ""
            und_val = (row.get(und_key) or "").strip() if und_key else ""
            cant_val = (row.get(cant_key) or "").strip() if cant_key else ""

            # Título del párrafo
            if item_val and desc_val:
                title = f"Ítem {item_val}: {desc_val}"
            elif desc_val:
                title = desc_val
            elif item_val:
                title = f"Ítem {item_val}"
            else:
                title = ""

            parts: List[str] = []
            if title:
                parts.append(title)

            # Detalles secundarios
            details: List[str] = []
            if und_val:
                details.append(f"Unidad: {und_val}")
            if cant_val:
                details.append(f"Cantidad: {cant_val}")

            for ek in extra_keys:
                v = (row.get(ek) or "").strip()
                if v:
                    details.append(f"{ek}: {v}")

            if details:
                parts.append(". ".join(details))

            para = ". ".join(parts)
            if para and not para.endswith("."):
                para += "."

            if para:
                paragraphs.append(para)

        header_text = "CARACTERÍSTICAS TÉCNICAS"
        if paragraphs:
            text = header_text + "\n\n" + "\n\n".join(paragraphs)
        else:
            text = segment
    else:
        text = segment

    max_len = 8000  # límite razonable
    if len(text) > max_len:
        text = text[:max_len] + "\n[...]"

    return text


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
                continue

        if not clicked:
            print("⚠️ Locator no pudo hacer clic; intentando DOM click via evaluate...")
            try:
                clicked = await page.evaluate(
                    """
                    () => {
                        const elements = Array.from(
                            document.querySelectorAll('button, input[type="button"], input[type="submit"]')
                        );
                        for (const el of elements) {
                            const text = (el.innerText || el.value || '').trim().toLowerCase();
                            if (text.includes('buscar')) {
                                el.click();
                                return true;
                            }
                        }
                        return false;
                    }
                    """
                )
                if clicked:
                    print("🔎 Click en botón Buscar (DOM evaluate).")
                else:
                    print("❌ No se encontró ningún botón 'Buscar' via DOM.")
            except Exception as e:
                print(f"❌ Error en DOM click de Buscar: {e}")
                clicked = False

        if not clicked:
            print("⚠️ No se logró hacer clic en Buscar. Dump de debug...")
            await debug_dump_page(page, label=f"no_search_button_attempt_{attempt}")
            return

        try:
            await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
        except Exception:
            pass

        await asyncio.sleep(3)
        await debug_dump_page(page, label=f"after_search_attempt_{attempt}")

    async def check_search_result_status(self, page: Page) -> str:
        """
        Clasifica el estado tras Buscar:
          - "ok": hay tabla con filas
          - "captcha_error": mensaje de error de captcha
          - "empty": sin filas válidas
        """
        status = await page.evaluate(
            """
            () => {
                function bodyHas(text) {
                    if (!document.body) return false;
                    return document.body.innerText.toLowerCase().includes(text.toLowerCase());
                }

                if (bodyHas("captcha incorrecto") ||
                    bodyHas("código captcha incorrecto") ||
                    bodyHas("codigo captcha incorrecto") ||
                    bodyHas("ingrese el código") ||
                    bodyHas("ingrese codigo") ||
                    bodyHas("código captcha") ) {
                    return "captcha_error";
                }

                let rows = Array.from(document.querySelectorAll("table tbody tr"));
                if (!rows.length) {
                    rows = Array.from(document.querySelectorAll("tbody tr"));
                }

                let dataRows = 0;
                for (const r of rows) {
                    const cells = r.querySelectorAll("td");
                    if (cells.length >= 3) {
                        const c0 = (cells[0].innerText || "").trim();
                        const c1 = (cells[1].innerText || "").trim();
                        if (c0 && c1) {
                            dataRows++;
                        }
                    }
                }

                if (dataRows > 0) return "ok";
                return "empty";
            }
            """
        )

        print(f"🔍 Estado detectado tras Buscar: {status}")
        return status

    async def solve_and_search_with_retries(self, page: Page) -> bool:
        """
        Reintenta resolver el CAPTCHA y hacer Buscar hasta que:
          - Se detecten filas de convocatorias ("ok"), o
          - Se agoten los max_captcha_attempts.
        """
        for attempt in range(1, self.max_captcha_attempts + 1):
            if attempt > 1:
                print("🔄 Re-cargando página para nuevo intento de CAPTCHA...")
                await page.goto(self.url, timeout=self.timeout_ms)
                await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                await self.handle_overlays(page)

            await self.fill_captcha_and_click_search_once(page, attempt)
            status = await self.check_search_result_status(page)

            if status == "ok":
                print("✅ Búsqueda con CAPTCHA exitosa, se encontró tabla con filas.")
                return True
            elif status == "captcha_error":
                print("⚠️ Error de CAPTCHA detectado, se reintentará si hay intentos disponibles.")
            else:
                print("ℹ️ No se detectaron filas válidas; intento fallido.")

        print("❌ No se logró pasar el CAPTCHA / obtener filas tras reintentos.")
        return False

    # -------------------------- Parsing fechas ------------------------------

    def parse_cierre_postulacion(self, text: str) -> Optional[datetime]:
        """
        Intenta extraer una fecha/hora de cierre.
        Devuelve datetime con tz America/Lima o None.
        """
        if not text:
            return None

        text = text.strip()
        if not text:
            return None

        # 1) Formatos tipo DD/MM/YYYY o DD-MM-YYYY (con o sin hora)
        m = re.search(
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?)",
            text,
        )
        if m:
            date_str = m.group(1)
            formatos = [
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M",
                "%d/%m/%Y",
                "%d-%m-%Y %H:%M:%S",
                "%d-%m-%Y %H:%M",
                "%d-%m-%Y",
            ]
            for fmt in formatos:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.replace(tzinfo=LIMA_TZ)
                except ValueError:
                    continue

        # 2) Formatos tipo "02 Dec 25" o "02 December 2025"
        m2 = re.search(
            r"(\d{1,2}\s+[A-Za-z]{3,}\s+\d{2,4})",
            text,
        )
        if m2:
            date_str2 = m2.group(1)
            formatos2 = [
                "%d %b %Y",
                "%d %b %y",
                "%d %B %Y",
                "%d %B %y",
            ]
            for fmt in formatos2:
                try:
                    dt = datetime.strptime(date_str2, fmt)
                    return dt.replace(tzinfo=LIMA_TZ)
                except ValueError:
                    continue

        return None

    def normalize_and_sort_convocatorias(
        self,
        rows: List[Dict[str, Any]],
        run_ts: datetime,
    ) -> List[Dict[str, Any]]:
        """
        - Convierte cierre_postulacion -> cierre_postulacion_lima (ISO con tz).
        - Ordena por fecha de cierre DESCENDENTE (la más lejana/reciente primero).
        - Estandariza fecha_extraccion en hora Lima.
        """
        for item in rows:
            raw_cierre = (item.get("cierre_postulacion") or "").strip()
            dt = self.parse_cierre_postulacion(raw_cierre)

            if dt is not None:
                item["cierre_postulacion_lima"] = dt.isoformat()
                item["_sort_cierre_dt"] = dt
            else:
                item["cierre_postulacion_lima"] = None
                item["_sort_cierre_dt"] = None

            item["fecha_extraccion"] = run_ts.isoformat()

        # Para orden descendente: los None al final -> asignamos datetime.min
        default_dt = datetime.min.replace(tzinfo=LIMA_TZ)
        sorted_rows = sorted(
            rows,
            key=lambda x: x.get("_sort_cierre_dt") or default_dt,
            reverse=True,  # DESCENDENTE
        )

        for item in sorted_rows:
            item.pop("_sort_cierre_dt", None)

        return sorted_rows

    # -------------------------- Extracción DOM ------------------------------

    async def extract_page(self, page: Page) -> List[Dict[str, Any]]:
        """
        Extrae filas de la página actual (sin descargar PDFs todavía).
        Añade _row_index_in_page para luego poder localizar la fila real
        y disparar la descarga desde Playwright.
        """
        print("📊 Extrayendo filas de la página actual...")
        try:
            await page.wait_for_selector("table, tbody tr", timeout=self.timeout_ms)
        except Exception:
            print("⚠️ No se encontraron tablas. Se devuelve lista vacía.")
            return []

        await self.handle_overlays(page)

        data: List[Dict[str, Any]] = await page.evaluate(
            """
            () => {
                const results = [];
                function clean(t) {
                    return t ? t.trim().replace(/\s+/g, ' ') : '';
                }

                let rows = Array.from(document.querySelectorAll('table tbody tr'));
                if (!rows.length) {
                    rows = Array.from(document.querySelectorAll('tbody tr'));
                }

                rows.forEach((row, idx) => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length < 4) return;

                    const item = {
                        numero_convocatoria: clean(cells[0]?.innerText || ''),
                        unidad_organica:    clean(cells[1]?.innerText || ''),
                        descripcion:        clean(cells[2]?.innerText || ''),
                        cierre_postulacion: clean(cells[3]?.innerText || ''),
                        _row_index_in_page: idx,
                    };

                    if (item.numero_convocatoria) {
                        results.push(item);
                    }
                });

                return results;
            }
            """
        )

        print(f"✅ Filas extraídas en esta página (sin TDR): {len(data)}")
        return data

    async def enrich_row_with_tdr_pdf(
        self,
        page: Page,
        row_locator,
        item: Dict[str, Any],
    ) -> None:
        """
        Para una fila concreta:
        - Intenta localizar el elemento clicable del TDR.
        - Usa expect_download() para capturar el PDF.
        - Aplica extracción + OCR y guarda todo en:
            * 'caracteristicas_tecnicas' -> texto ordenado en párrafos.
        """
        try:
            clickable = row_locator.locator("a, button, img, span")
            n = await clickable.count()
            if n == 0:
                return

            candidate = None

            for j in range(n):
                el = clickable.nth(j)
                try:
                    txt = (await el.inner_text()).strip().lower()
                except Exception:
                    txt = ""

                alt = (await el.get_attribute("alt")) or ""
                title = (await el.get_attribute("title")) or ""
                src = (await el.get_attribute("src")) or ""
                onclick = (await el.get_attribute("onclick")) or ""

                combined = " ".join([txt, alt, title, onclick]).lower()

                # Evitar el típico "Ver" genérico sin contexto de TDR
                if "ver" in txt and "pdf" not in combined and "tdr" not in combined:
                    continue

                if any(
                    kw in combined
                    for kw in [
                        "tdr",
                        "especificacion",
                        "especificación",
                        "caracteristica tecnica",
                        "característica técnica",
                    ]
                ) or "pdf" in src.lower():
                    candidate = el
                    break

            if candidate is None:
                # No se encontró botón/ícono razonable para TDR
                return

            print(f"📥 Intentando descargar TDR para {item.get('numero_convocatoria')}...")
            try:
                async with page.expect_download(timeout=self.timeout_ms) as dl_info:
                    await candidate.click()
                download = await dl_info.value
            except Exception as e:
                print(f"⚠️ No se produjo descarga para la fila: {e}")
                return

            tmp_path = await download.path()
            if not tmp_path:
                print("⚠️ Descarga sin ruta de archivo (tmp_path vacío).")
                return

            suggested_name = download.suggested_filename or os.path.basename(tmp_path)
            item["tdr_filename"] = suggested_name
            item["tdr_downloaded"] = True

            # Extraer bloque de características como texto ordenado
            block = extract_caracteristicas_from_pdf(tmp_path)
            item["caracteristicas_tecnicas"] = block

        except Exception as e:
            print(f"⚠️ Error enriqueciendo fila con TDR: {e}")

    async def paginate_and_extract(self, page: Page) -> List[Dict[str, Any]]:
        """
        Recorre las páginas usando "Siguiente", acumulando filas.
        Para cada fila intenta descargar y procesar el TDR.
        """
        all_rows: List[Dict[str, Any]] = []

        for idx_page in range(self.max_pages):
            print(f"📄 Página {idx_page + 1}/{self.max_pages}")
            page_rows = await self.extract_page(page)

            # Locator de filas reales en la página para mapear _row_index_in_page
            rows_locator = page.locator("table tbody tr")
            cnt = await rows_locator.count()
            if cnt == 0:
                rows_locator = page.locator("tbody tr")
                cnt = await rows_locator.count()

            for item in page_rows:
                row_idx = item.get("_row_index_in_page")
                if row_idx is None or row_idx >= cnt:
                    continue

                row_loc = rows_locator.nth(row_idx)
                await self.enrich_row_with_tdr_pdf(page, row_loc, item)
                item.pop("_row_index_in_page", None)
                all_rows.append(item)

            # Buscar botón "Siguiente"
            next_selectors = [
                'a[aria-label*="Siguiente"]',
                'button[aria-label*="Siguiente"]',
                'a:has-text("Siguiente")',
                'button:has-text("Siguiente")',
                'a:has-text("Sig.")',
                'a[title*="Siguiente"]',
            ]
            next_btn = None
            for sel in next_selectors:
                try:
                    cand = page.locator(sel).first
                    if await cand.is_visible(timeout=2000):
                        aria_dis = await cand.get_attribute("aria-disabled")
                        disabled = await cand.get_attribute("disabled")
                        if aria_dis in ("true", "1") or disabled is not None:
                            continue
                        next_btn = cand
                        break
                except Exception:
                    continue

            if not next_btn:
                print("⏹️ No se encontró 'Siguiente' activo. Fin de paginación.")
                break

            print("➡️ Avanzando a la siguiente página...")
            try:
                await next_btn.click()
            except Exception as e:
                print(f"⚠️ Error haciendo clic en Siguiente: {e}")
                break

            try:
                await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                await asyncio.sleep(1)
            except Exception:
                pass

        print(f"📊 Total filas acumuladas (sin normalizar): {len(all_rows)}")
        return all_rows

    # -------------------------- Ejecución ----------------------------------

    async def run(self) -> Dict[str, Any]:
        print("🚀 Iniciando scraper PJ 8UIT...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            page = await browser.new_page()
            await self.setup_page(page)

            try:
                print(f"🌐 Navegando a: {self.url}")
                await page.goto(self.url, timeout=self.timeout_ms)
                await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                await self.handle_overlays(page)

                ok = await self.solve_and_search_with_retries(page)
                if not ok:
                    print("❌ No se logró obtener tabla con filas tras reintentos.")
                    now_lima = datetime.now(LIMA_TZ)
                    return {
                        "success": False,
                        "timestamp": now_lima.isoformat(),
                        "source_url": self.url,
                        "total_convocatorias": 0,
                        "convocatorias": [],
                        "error": "No se logró pasar CAPTCHA / obtener filas tras reintentos",
                    }

                rows_raw = await self.paginate_and_extract(page)
                run_ts = datetime.now(LIMA_TZ)
                rows = self.normalize_and_sort_convocatorias(rows_raw, run_ts)

                result: Dict[str, Any] = {
                    "success": True,
                    "timestamp": run_ts.isoformat(),
                    "source_url": self.url,
                    "total_convocatorias": len(rows),
                    "convocatorias": rows,
                }
                print(f"🎉 Scraping terminado. Total convocatorias: {len(rows)}")
                return result

            finally:
                await browser.close()


# ---------------------------------------------------------------------------
# Wrappers para GitHub Actions y pruebas locales
# ---------------------------------------------------------------------------

async def run_github() -> int:
    captcha_code = os.getenv("CAPTCHA_CODE") or os.getenv("PJ_CAPTCHA")
    timeout_env = os.getenv("SCRAPER_TIMEOUT", "60")
    try:
        timeout_val = int(timeout_env)
    except ValueError:
        timeout_val = 60

    scraper = PJScraper(
        headless=True,
        timeout=timeout_val,
        captcha_code=captcha_code,
        use_tesseract=True,
        max_captcha_attempts=5,
    )
    result = await scraper.run()

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    ts = datetime.now(LIMA_TZ).strftime("%Y%m%d_%H%M%S")

    raw_file = f"data/raw/pj_8uit_raw_{ts}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    convocatorias = result.get("convocatorias", [])
    total = result.get("total_convocatorias", len(convocatorias))

    processed = {
        "metadata": {
            "source": "pj_8uit_convocatorias",
            "extraction_timestamp": result.get(
                "timestamp",
                datetime.now(LIMA_TZ).isoformat()
            ),
            "total_records": total,
        },
        "convocatorias": convocatorias,
    }
    proc_file = f"data/processed/pj_8uit_tdr_{ts}.json"
    with open(proc_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print("📁 Archivos guardados:")
    print(f"  RAW JSON:  {raw_file}")
    print(f"  PROC JSON: {proc_file}")

    return 0 if result.get("success") else 1


async def run_local() -> None:
    """Modo debug local (abre navegador visible)."""
    scraper = PJScraper(
        headless=False,
        timeout=90,
        captcha_code=None,
        use_tesseract=True,
        max_captcha_attempts=5,
    )
    result = await scraper.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "github":
        code = asyncio.run(run_github())
        sys.exit(code)
    else:
        asyncio.run(run_local())
