#!/usr/bin/env python3
"""
PJ 8UIT Scraper

Flujo general:
- Abrir https://sap.pj.gob.pe/portalabastecimiento-web/Convocatorias8uit
- Resolver CAPTCHA (Tesseract OCR o código fijo por env var).
- Escribir el CAPTCHA en el input y hacer clic en "Buscar".
- Reintentar si el CAPTCHA es incorrecto o no hay filas.
- Una vez cargada la tabla:
    * Recorrer todas las páginas de resultados (paginador con iconos "Next").
    * Extraer, por fila:
        - numero_convocatoria
        - unidad_organica
        - descripcion
        - cierre_postulacion (texto original)
    * Intentar localizar el botón/ícono de TDR y descargar el PDF.
    * Extraer el bloque "CARACTERISTICAS TECNICAS" del PDF (si tiene texto).
- Normalizar fechas a zona horaria America/Lima y ordenar por cierre descendente.
- En modo GitHub:
    * Guardar un único JSON procesado en data/processed/pj_8uit_tdr_YYYYMMDD_HHMMSS.json
"""

import asyncio
import io
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    - Devuelve el mejor candidato (prioriza longitud 4–6).
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
    for desc, txt in attempts:
        if 4 <= len(txt) <= 6 and not txt.startswith("ERROR:"):
            best = txt
            break

    if not best:
        candidates = [txt for _, txt in attempts if txt and not txt.startswith("ERROR:")]
        if candidates:
            best = max(candidates, key=len)

    if best:
        print(f"✅ Mejor resultado OCR: '{best}'")
        return best

    print("❌ No se obtuvo un resultado OCR usable.")
    return None


def extract_caracteristicas_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extrae el bloque de texto correspondiente a "CARACTERISTICAS TECNICAS"
    (tolerando variantes con tilde) desde el PDF. Si no hay texto (PDF escaneado),
    devolverá None.
    """
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"⚠️ No se pudo abrir PDF '{pdf_path}': {e}")
        return None

    texts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            texts.append(t)

    if not texts:
        print(f"⚠️ PDF sin texto legible: {pdf_path}")
        return None

    full_text = "\n".join(texts)
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
    max_len = 4000
    if len(segment) > max_len:
        segment = segment[:max_len] + "\n[...]"

    return segment


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
        Intenta extraer una fecha/hora de cierre y devolver datetime con tz America/Lima.
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
        - Ordena por fecha de cierre DESCENDENTE.
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

        default_dt = datetime.min.replace(tzinfo=LIMA_TZ)
        sorted_rows = sorted(
            rows,
            key=lambda x: x.get("_sort_cierre_dt") or default_dt,
            reverse=True,
        )

        for item in sorted_rows:
            item.pop("_sort_cierre_dt", None)

        return sorted_rows

    # -------------------------- Extracción DOM ------------------------------

    async def extract_page(self, page: Page) -> List[Dict[str, Any]]:
        """
        Extrae filas de la página actual (sin descargar PDFs todavía).
        Añade _row_index_in_page para luego poder localizar la fila real.
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
                    return t ? t.trim().replace(/\\s+/g, ' ') : '';
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
        - Extrae 'caracteristicas_tecnicas' del PDF (solo si tiene texto).
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

            block = extract_caracteristicas_from_pdf(tmp_path)
            if block:
                item["caracteristicas_tecnicas"] = block
            else:
                item["caracteristicas_tecnicas"] = None

        except Exception as e:
            print(f"⚠️ Error enriqueciendo fila con TDR: {e}")

    async def paginate_and_extract(self, page: Page) -> List[Dict[str, Any]]:
        """
        Recorre las páginas usando el paginador (iconos 'Siguiente'), acumulando filas.
        Para cada fila intenta descargar y procesar el TDR.
        """
        all_rows: List[Dict[str, Any]] = []
        last_first_key: Optional[str] = None

        for idx_page in range(self.max_pages):
            print(f"📄 Página {idx_page + 1}/{self.max_pages}")

            page_rows = await self.extract_page(page)
            if not page_rows:
                print("⚠️ Sin filas en esta página. Fin de paginación.")
                break

            first_key = (page_rows[0].get("numero_convocatoria") or "").strip()
            if last_first_key is not None and first_key == last_first_key:
                print("⏹️ La primera fila es igual a la de la página anterior. Se asume fin de páginas.")
                break

            last_first_key = first_key

            rows_locator = page.lo_
