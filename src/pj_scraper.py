#!/usr/bin/env python3
"""
Scraper PJ 8UIT con:
- Resolución de CAPTCHA vía Tesseract (OCR).
- Click en "Buscar" con reintentos.
- Paginación de resultados.
- Descarga de TDR (PDF) por convocatoria.
- Extracción de sección "CARACTERÍSTICAS / ESPECIFICACIONES TÉCNICAS"
  (texto + OCR si el PDF es imagen).
- Salida ÚNICA: JSON por stdout (para GitHub / n8n).
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
from PIL import Image, ImageFilter
import pytesseract
from PyPDF2 import PdfReader

# URL de la página de convocatorias < 8 UIT
URL = "https://sap.pj.gob.pe/portalabastecimiento-web/Convocatorias8uit"
LIMA_TZ = ZoneInfo("America/Lima")


# --------------------------------------------------------------------
# Utilidades OCR / PDFs
# --------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Normaliza texto OCR a A-Z / 0-9 en mayúsculas."""
    return "".join(c for c in text if c.isalnum()).upper()


def solve_captcha_tesseract_advanced(image: Image.Image) -> Optional[str]:
    """
    Resuelve el CAPTCHA probando varias configuraciones de Tesseract
    y versiones reescaladas de la imagen.
    """
    configs: List[Tuple[str, str]] = [
        ("psm8_whitelist", "--oem 3 --psm 8 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("psm7_whitelist", "--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ]

    variants: List[Tuple[str, Image.Image]] = []
    base_gray = image.convert("L")
    variants.append(("gray", base_gray))

    # Reescalar para mejorar OCR si el CAPTCHA es pequeño
    for scale in (2, 3):
        w, h = base_gray.size
        resized = base_gray.resize((w * scale, h * scale))
        variants.append((f"gray_x{scale}", resized))

    attempts: List[Tuple[str, str]] = []

    for v_name, img_v in variants:
        for cfg_name, cfg in configs:
            desc = f"{v_name}|{cfg_name}"
            try:
                raw = pytesseract.image_to_string(img_v, config=cfg)
                clean = _clean_text(raw)
                attempts.append((desc, clean))
            except Exception as e:
                attempts.append((desc, f"ERROR:{e}"))

    best = ""
    # Primer filtro: 4–6 caracteres, parece típico CAPTCHA
    for _, txt in attempts:
        if 4 <= len(txt) <= 6 and not txt.startswith("ERROR:"):
            best = txt
            break

    # Si no hay nada con 4–6 chars, tomar el más largo que no sea ERROR
    if not best:
        cands = [t for _, t in attempts if t and not t.startswith("ERROR:")]
        if cands:
            best = max(cands, key=len)

    return best or None


def extract_text_from_pdf_with_ocr(pdf_path: str) -> str:
    """
    Extrae texto de un PDF:
    1) Intenta primero con PyPDF2 (texto embebido).
    2) Si el texto es muy corto, usa OCR con pdftoppm + Tesseract.
    """
    text_parts: List[str] = []

    # 1) Intento con PyPDF2
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    except Exception as e:
        print("PyPDF2 error:", e)

    text_py = "\n".join(text_parts).strip()

    # Si ya tenemos un bloque razonable, lo usamos
    if len(text_py) > 500:
        return text_py

    # 2) OCR con pdftoppm -> PNG -> Tesseract
    ocr_parts: List[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "page")
            # Convierte PDF a imágenes PNG
            subprocess.run(
                ["pdftoppm", "-png", pdf_path, out_prefix],
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
                        ocr_parts.append(txt)
                except Exception as e:
                    print("OCR page error", png, e)

    except Exception as e:
        print("pdftoppm/OCR pipeline error:", e)

    text_ocr = "\n".join(ocr_parts).strip()

    # Prioridad: texto OCR si existe, luego PyPDF2
    return text_ocr or text_py


def parse_caracteristicas_table(segment: str) -> Optional[List[Dict[str, str]]]:
    """
    Intenta detectar una tabla de características técnicas / especificaciones
    a partir de texto plano (segmento ya recortado).
    Devuelve lista de dicts (una por fila).
    """
    lines = [ln.strip() for ln in segment.splitlines() if ln.strip()]
    if not lines:
        return None

    header_idx = -1
    header_cols: List[str] = []

    # Detectar fila de cabecera por palabras clave
    for i, ln in enumerate(lines):
        up = ln.upper()
        if ("ITEM" in up or "ÍTEM" in up) and ("ESPECIFIC" in up or "DESCRIPC" in up):
            header_idx = i
            header_cols = re.split(r"\s{2,}", ln)
            header_cols = [c.strip() for c in header_cols if c.strip()]
            break

    if header_idx == -1 or not header_cols:
        return None

    rows: List[Dict[str, str]] = []

    for ln in lines[header_idx + 1 :]:
        up = ln.upper()
        # Corte cuando ya empiezan secciones de condiciones / requisitos, etc.
        if any(
            kw in up
            for kw in [
                "CONDICIONES",
                "REQUISITOS",
                "OBLIGACIONES",
                "PLAZO DE",
                "GARANT",
                "FORMA DE PAGO",
            ]
        ):
            break

        parts = re.split(r"\s{2,}", ln)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue

        # Ajustar columnas
        if len(parts) < len(header_cols):
            parts += [""] * (len(header_cols) - len(parts))
        if len(parts) > len(header_cols):
            parts = parts[: len(header_cols) - 1] + [" ".join(parts[len(header_cols) - 1 :])]

        row = {header_cols[i]: parts[i] for i in range(len(header_cols))}
        rows.append(row)

    return rows or None


def extract_caracteristicas_from_pdf(pdf_path: str) -> str:
    """
    Extrae la sección de CARACTERÍSTICAS / ESPECIFICACIONES TÉCNICAS del PDF.
    Si no se encuentra, devuelve un resumen del texto.
    """
    full = extract_text_from_pdf_with_ocr(pdf_path)
    if not full:
        return "No fue posible extraer texto útil del TDR."

    norm = full.upper()

    # Buscar títulos típicos
    start = -1
    for pat in [
        r"CARACTER[IÍ]STICAS\s+T[ÉE]CNICAS",
        r"ESPECIFICACIONES\s+T[ÉE]CNICAS",
    ]:
        m = re.search(pat, norm)
        if m:
            start = m.start()
            break

    # Si no se encuentra sección, devolver un recorte grande del texto
    if start == -1:
        return full[:8000]

    # Determinar fin aproximado de la sección por marcadores
    end = len(full)
    for mark in [
        "CONDICIONES GENERALES",
        "CONDICIONES",
        "REQUISITOS",
        "OBLIGACIONES",
        "PLAZO DE",
        "GARANT",
        "FORMA DE PAGO",
    ]:
        pos = norm.find(mark, start + 5)
        if pos != -1 and pos > start:
            end = min(end, pos)

    segment = full[start:end].strip()

    # Intentar reconstruir tabla
    tabla = parse_caracteristicas_table(segment)
    if not tabla:
        return segment[:8000]

    headers = list(tabla[0].keys())

    def find_key(pred):
        for k in headers:
            if pred(k.upper()):
                return k
        return None

    item_key = find_key(lambda u: "ITEM" in u or "ÍTEM" in u)
    desc_key = find_key(lambda u: "ESPECIFIC" in u or "DESCRIPC" in u)
    und_key = find_key(lambda u: "UND" in u or "UNIDAD" in u)
    cant_key = find_key(lambda u: "CANT" in u)

    extra_keys = [
        k
        for k in headers
        if k not in {item_key, desc_key, und_key, cant_key} and k is not None
    ]

    def sort_key(row):
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

    ordered = sorted(tabla, key=sort_key)

    paras: List[str] = []
    for row in ordered:
        item = (row.get(item_key) or "").strip() if item_key else ""
        desc = (row.get(desc_key) or "").strip() if desc_key else ""
        und = (row.get(und_key) or "").strip() if und_key else ""
        cant = (row.get(cant_key) or "").strip() if cant_key else ""

        if item and desc:
            title = f"Ítem {item}: {desc}"
        elif desc:
            title = desc
        elif item:
            title = f"Ítem {item}"
        else:
            title = ""

        details: List[str] = []
        if und:
            details.append(f"Unidad: {und}")
        if cant:
            details.append(f"Cantidad: {cant}")
        for ek in extra_keys:
            v = (row.get(ek) or "").strip()
            if v:
                details.append(f"{ek}: {v}")

        para = title
        if details:
            if para:
                para += ". "
            para += ". ".join(details)
        if para and not para.endswith("."):
            para += "."

        if para:
            paras.append(para)

    text = "CARACTERÍSTICAS / ESPECIFICACIONES TÉCNICAS\n\n" + "\n\n".join(paras)
    return text[:8000]


# --------------------------------------------------------------------
# Scraper principal
# --------------------------------------------------------------------
class PJScraper:
    def __init__(
        self,
        headless: bool = True,
        timeout: int = 60,
        captcha_code: Optional[str] = None,
        max_pages: int = 30,
        max_captcha_attempts: int = 5,
        use_tesseract: bool = True,
    ):
        self.url = URL
        self.headless = headless
        self.timeout_ms = timeout * 1000
        self.captcha_code = captcha_code
        self.max_pages = max_pages
        self.max_captcha_attempts = max_captcha_attempts
        self.use_tesseract = use_tesseract

        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-PE,es;q=0.9,en;q=0.8",
        }

    async def setup_page(self, page: Page):
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.set_extra_http_headers(self.headers)
        page.on("popup", lambda p: asyncio.create_task(p.close()))
        page.on("dialog", lambda d: asyncio.create_task(d.accept()))
        await page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        )

    async def handle_overlays(self, page: Page):
        """Cerrar modales / overlays molestos si aparecen."""
        try:
            modal_sel = [
                ".modal:visible",
                ".popup:visible",
                ".dialog:visible",
                '[role="dialog"]:visible',
            ]
            for sel in modal_sel:
                try:
                    loc = page.locator(sel).first
                    if await loc.is_visible(timeout=2000):
                        await page.keyboard.press("Escape")
                        break
                except Exception:
                    continue
        except Exception as e:
            print("overlays error:", e)

    async def locate_captcha_img(self, page: Page):
        sels = [
            'img[src*="captcha"]',
            'img[alt*="captcha"]',
            'img[id*="captcha"]',
            ".captcha img",
            "#captcha img",
            'img[src*="Codigo"]',
            'img[src*="codigo"]',
        ]
        for sel in sels:
            try:
                img = page.locator(sel).first
                if await img.is_visible(timeout=3000):
                    return img
            except Exception:
                continue
        return None

    async def has_captcha(self, page: Page) -> bool:
        return (await self.locate_captcha_img(page)) is not None

    async def capture_captcha_image(self, page: Page) -> Optional[Image.Image]:
        """Captura el CAPTCHA recortando la screenshot de la página."""
        img = await self.locate_captcha_img(page)
        if img is None:
            return None

        bbox = await img.bounding_box()
        if not bbox:
            return None

        screenshot = await page.screenshot()
        full = Image.open(io.BytesIO(screenshot))

        pad = 4
        left = max(0, int(bbox["x"] - pad))
        top = max(0, int(bbox["y"] - pad))
        right = min(full.width, int(bbox["x"] + bbox["width"] + pad))
        bottom = min(full.height, int(bbox["y"] + bbox["height"] + pad))

        captcha = full.crop((left, top, right, bottom))

        os.makedirs("captcha_images", exist_ok=True)
        ts = datetime.now(LIMA_TZ).strftime("%Y%m%d_%H%M%S")
        captcha.save(f"captcha_images/captcha_raw_{ts}.png")

        return captcha

    async def fill_captcha_and_click_search_once(self, page: Page, attempt: int):
        """
        Rellena el CAPTCHA (si está presente) y hace un click en "Buscar".
        """
        if await self.has_captcha(page):
            text = None
            if self.captcha_code:
                text = self.captcha_code.strip()
            elif self.use_tesseract:
                img = await self.capture_captcha_image(page)
                if img:
                    text = solve_captcha_tesseract_advanced(img)

            input_loc = None
            for sel in [
                'input[name*="captcha"]',
                'input[id*="captcha"]',
                'input[placeholder*="aptcha"]',
                "input[type='text']",
            ]:
                try:
                    cand = page.locator(sel).first
                    if await cand.is_visible(timeout=2000):
                        input_loc = cand
                        break
                except Exception:
                    continue

            if input_loc and text:
                await input_loc.fill("")
                await input_loc.type(text, delay=80)

        clicked = False
        for sel in [
            'button:has-text("Buscar")',
            'input[value*="Buscar"]',
            'input[type="submit"]',
            'button[type="submit"]',
        ]:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=2000):
                    await btn.click()
                    clicked = True
                    break
            except Exception:
                continue

        if not clicked:
            try:
                clicked = await page.evaluate(
                    """
                () => {
                  const els=[...document.querySelectorAll('button,input[type="button"],input[type="submit"]')];
                  for(const el of els){
                    const t=(el.innerText||el.value||'').trim().toLowerCase();
                    if(t.includes('buscar')){ el.click(); return true;}
                  }
                  return false;
                }"""
                )
            except Exception as e:
                print("DOM click error:", e)

        await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
        await asyncio.sleep(3)

    async def check_search_result_status(self, page: Page) -> str:
        """
        Devuelve:
        - "captcha_error" si se detecta mensaje de error de captcha.
        - "ok" si hay filas con datos en la tabla.
        - "empty" si no hay datos.
        """
        status = await page.evaluate(
            """
        () => {
          function bodyHas(txt){
            return document.body && document.body.innerText.toLowerCase().includes(txt.toLowerCase());
          }
          if (bodyHas("captcha incorrecto") || bodyHas("codigo captcha incorrecto")) return "captcha_error";
          let rows=[...document.querySelectorAll("table tbody tr")];
          if(!rows.length) rows=[...document.querySelectorAll("tbody tr")];
          let data=0;
          for(const r of rows){
            const tds=r.querySelectorAll("td");
            if(tds.length>=3){
              const a=(tds[0].innerText||'').trim();
              const b=(tds[1].innerText||'').trim();
              if(a&&b) data++;
            }
          }
          if(data>0) return "ok";
          return "empty";
        }"""
        )
        return status

    async def solve_and_search_with_retries(self, page: Page) -> bool:
        """
        Reintenta varias veces hasta conseguir resultados en la tabla.
        """
        for attempt in range(1, self.max_captcha_attempts + 1):
            if attempt > 1:
                await page.goto(self.url, timeout=self.timeout_ms)
                await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                await self.handle_overlays(page)

            await self.fill_captcha_and_click_search_once(page, attempt)
            status = await self.check_search_result_status(page)

            if status == "ok":
                return True

        return False

    def parse_cierre_postulacion(self, text: str) -> Optional[datetime]:
        """Convierte la fecha de cierre a datetime en zona Lima."""
        if not text:
            return None
        text = text.strip()
        m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", text)
        if m:
            for fmt in ["%d/%m/%Y", "%d/%m/%y"]:
                try:
                    dt = datetime.strptime(m.group(1), fmt)
                    return dt.replace(tzinfo=LIMA_TZ)
                except ValueError:
                    continue
        return None

    def normalize_and_sort(
        self, rows: List[Dict[str, Any]], run_ts: datetime
    ) -> List[Dict[str, Any]]:
        """Normaliza fecha y ordena de más reciente a más antigua."""
        for it in rows:
            raw = (it.get("cierre_postulacion") or "").strip()
            dt = self.parse_cierre_postulacion(raw)
            it["cierre_postulacion_lima"] = dt.isoformat() if dt else None
            it["_sort_dt"] = dt or datetime.min.replace(tzinfo=LIMA_TZ)
            it["fecha_extraccion"] = run_ts.isoformat()

        rows_sorted = sorted(rows, key=lambda x: x["_sort_dt"], reverse=True)
        for it in rows_sorted:
            it.pop("_sort_dt", None)
        return rows_sorted

    async def extract_page(self, page: Page) -> List[Dict[str, Any]]:
        """Extrae las filas de la página actual (solo datos de la tabla)."""
        await page.wait_for_selector("table,tbody tr", timeout=self.timeout_ms)
        await self.handle_overlays(page)

        data: List[Dict[str, Any]] = await page.evaluate(
            """
        () => {
          const res=[];
          const clean=t=>t?t.trim().replace(/\\s+/g,' '):'';
          let rows=[...document.querySelectorAll('table tbody tr')];
          if(!rows.length) rows=[...document.querySelectorAll('tbody tr')];
          rows.forEach((row,idx)=>{
            const tds=row.querySelectorAll('td');
            if(tds.length<4) return;
            const item={
              numero_convocatoria:clean(tds[0].innerText),
              unidad_organica:clean(tds[1].innerText),
              descripcion:clean(tds[2].innerText),
              cierre_postulacion:clean(tds[3].innerText),
              _row_index_in_page:idx,
            };
            if(item.numero_convocatoria) res.push(item);
          });
          return res;
        }"""
        )

        return data

    async def enrich_row_with_tdr(self, page: Page, row_locator, item: Dict[str, Any]):
        """
        En una fila de la tabla, intenta localizar y descargar el TDR (PDF),
        luego extraer la sección de características técnicas.
        """
        try:
            clickable = row_locator.locator("a,button,img,span")
            n = await clickable.count()
            if n == 0:
                item["tdr_downloaded"] = False
                item["caracteristicas_tecnicas"] = "No se encontró TDR en la fila."
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
                if "pdf" in src.lower() or "tdr" in combined or "especificacion" in combined:
                    candidate = el
                    break

            if candidate is None:
                item["tdr_downloaded"] = False
                item["caracteristicas_tecnicas"] = "No se identificó link/ícono TDR."
                return

            async with page.expect_download(timeout=self.timeout_ms) as dl_info:
                await candidate.click()
            download = await dl_info.value

            tmp_path = await download.path()
            if not tmp_path:
                item["tdr_downloaded"] = False
                item["caracteristicas_tecnicas"] = "Descarga sin ruta TDR."
                return

            item["tdr_filename"] = download.suggested_filename or os.path.basename(tmp_path)
            item["tdr_downloaded"] = True

            block = extract_caracteristicas_from_pdf(tmp_path)
            item["caracteristicas_tecnicas"] = block

        except Exception as e:
            item["tdr_downloaded"] = False
            item["caracteristicas_tecnicas"] = f"Error al procesar TDR: {e}"

    async def paginate_and_extract(self, page: Page) -> List[Dict[str, Any]]:
        """Recorre páginas (Siguiente) y enriquece cada fila con TDR."""
        all_rows: List[Dict[str, Any]] = []

        for page_idx in range(self.max_pages):
            page_rows = await self.extract_page(page)

            rows_locator = page.locator("table tbody tr")
            cnt = await rows_locator.count()
            if cnt == 0:
                rows_locator = page.locator("tbody tr")
                cnt = await rows_locator.count()

            for item in page_rows:
                row_idx = item.get("_row_index_in_page")
                if row_idx is None or row_idx >= cnt:
                    item.pop("_row_index_in_page", None)
                    item["tdr_downloaded"] = False
                    item["caracteristicas_tecnicas"] = "No se pudo mapear fila DOM TDR."
                    all_rows.append(item)
                    continue

                row_loc = rows_locator.nth(row_idx)
                await self.enrich_row_with_tdr(page, row_loc, item)
                item.pop("_row_index_in_page", None)
                all_rows.append(item)

            # Intentar ir a la página siguiente
            next_btn = None
            for sel in [
                'a[aria-label*="Siguiente"]',
                'button[aria-label*="Siguiente"]',
                'a:has-text("Siguiente")',
                'button:has-text("Siguiente")',
            ]:
                try:
                    cand = page.locator(sel).first
                    if await cand.is_visible(timeout=2000):
                        aria = await cand.get_attribute("aria-disabled")
                        dis = await cand.get_attribute("disabled")
                        if aria in ("true", "1") or dis is not None:
                            continue
                        next_btn = cand
                        break
                except Exception:
                    continue

            if not next_btn:
                break

            try:
                await next_btn.click()
                await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                await asyncio.sleep(1)
            except Exception:
                break

        return all_rows

    async def run(self) -> Dict[str, Any]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = await browser.new_page()
            await self.setup_page(page)

            try:
                await page.goto(self.url, timeout=self.timeout_ms)
                await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                await self.handle_overlays(page)

                ok = await self.solve_and_search_with_retries(page)
                if not ok:
                    now = datetime.now(LIMA_TZ)
                    return {
                        "success": False,
                        "timestamp": now.isoformat(),
                        "source_url": self.url,
                        "total_convocatorias": 0,
                        "convocatorias": [],
                        "error": "No se logró pasar CAPTCHA/obtener filas",
                    }

                rows_raw = await self.paginate_and_extract(page)
                run_ts = datetime.now(LIMA_TZ)
                rows = self.normalize_and_sort(rows_raw, run_ts)

                return {
                    "success": True,
                    "timestamp": run_ts.isoformat(),
                    "source_url": self.url,
                    "total_convocatorias": len(rows),
                    "convocatorias": rows,
                }

            finally:
                await browser.close()


# --------------------------------------------------------------------
# Salida JSON única (para GitHub / n8n)
# --------------------------------------------------------------------
def build_processed_json(result: Dict[str, Any]) -> Dict[str, Any]:
    convs = result.get("convocatorias", [])
    total = result.get("total_convocatorias", len(convs))
    return {
        "metadata": {
            "source": "pj_8uit_convocatorias",
            "extraction_timestamp": result.get(
                "timestamp", datetime.now(LIMA_TZ).isoformat()
            ),
            "total_records": total,
        },
        "convocatorias": convs,
    }


async def run_github() -> int:
    """
    Entrada para GitHub Actions:
    - Lee SCRAPER_TIMEOUT y CAPTCHA_CODE de env.
    - Imprime SOLO el JSON final a stdout.
    """
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
    processed = build_processed_json(result)
    print(json.dumps(processed, indent=2, ensure_ascii=False))

    return 0 if result.get("success") else 1


async def run_local():
    """
    Ejecución local (para debug).
    """
    scraper = PJScraper(
        headless=False,
        timeout=90,
        captcha_code=None,
        use_tesseract=True,
        max_captcha_attempts=5,
    )
    result = await scraper.run()
    processed = build_processed_json(result)
    print(json.dumps(processed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Modo GitHub: `python src/pj_scraper.py github`
    if len(sys.argv) > 1 and sys.argv[1] == "github":
        code = asyncio.run(run_github())
        sys.exit(code)
    else:
        # Modo local (debug)
        asyncio.run(run_local())
