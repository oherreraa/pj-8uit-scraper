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
    * Guardar resultados en JSON (raw + procesado) para GitHub Actions.
"""

import asyncio
import json
import os
import sys
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Page

from PIL import Image, ImageOps, ImageFilter
import pytesseract


URL = "https://sap.pj.gob.pe/portalabastecimiento-web/Convocatorias8uit"


# ---------------------------------------------------------------------------
# Utilidades de OCR / debug
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Limpia el texto devuelto por Tesseract."""
    return "".join(c for c in text if c.isalnum()).upper()


def solve_captcha_tesseract_advanced(image: Image.Image) -> Optional[str]:
    """
    Versión avanzada de OCR:
    - Diversos preprocesados.
    - Varias configuraciones Tesseract.
    - Elige la mejor cadena según heurística simple.
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
        # Si no hay nada de longitud adecuada, elegimos la no vacía más larga
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


async def debug_dump_page(page: Page, label: str = "after_search") -> None:
    """
    Guardar screenshot + HTML de la página para debug.
    Útil para ver qué mensaje devuelve el portal cuando falla el CAPTCHA.
    """
    try:
        os.makedirs("data/raw", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        """Cerrar modales y aceptar cookies si aparecen."""
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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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

            # 1) Código fijo vía entorno / secrets
            if self.captcha_code:
                captcha_text = self.captcha_code.strip()
                print(f"🔑 Usando CAPTCHA fijo: '{captcha_text}'")
            # 2) OCR avanzado con Tesseract
            elif self.use_tesseract:
                img = await self.capture_captcha_image(page)
                if img is not None:
                    captcha_text = solve_captcha_tesseract_advanced(img)

            if not captcha_text:
                print(
                    "⚠️ No se obtuvo texto de CAPTCHA. "
                    "Es probable que la búsqueda falle."
                )

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

        # Botón Buscar (primer intento con Playwright)
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

        # Fallback: clic por DOM con evaluate en cualquier botón/input con texto "Buscar"
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

        # Esperar a que se procese la búsqueda
        try:
            await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
        except Exception:
            pass

        # Pequeña espera adicional para que cargue la tabla vía JS/XHR
        await asyncio.sleep(3)

        # Dump de la página tras este intento (para análisis)
        await debug_dump_page(page, label=f"after_search_attempt_{attempt}")

    async def check_search_result_status(self, page: Page) -> str:
        """
        Analiza el DOM después de Buscar y clasifica el estado:
          - "ok": hay tabla con filas de convocatorias
          - "captcha_error": mensaje típico de error de captcha
          - "empty": sin filas válidas (tabla vacía u otro mensaje)
        """
        status = await page.evaluate(
            """
            () => {
                function bodyHas(text) {
                    if (!document.body) return false;
                    return document.body.innerText.toLowerCase().includes(text.toLowerCase());
                }

                // Mensajes típicos de error de captcha
                if (bodyHas("captcha incorrecto") ||
                    bodyHas("código captcha incorrecto") ||
                    bodyHas("codigo captcha incorrecto") ||
                    bodyHas("ingrese el código") ||
                    bodyHas("ingrese codigo") ||
                    bodyHas("código captcha") ) {
                    return "captcha_error";
                }

                // Contar filas con datos en alguna tabla
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
        En cada intento se recarga la página para forzar nuevo CAPTCHA.
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
                # "empty": puede ser tabla vacía o mensaje extraño
                print("ℹ️ No se detectaron filas válidas; se considera intento fallido.")

        print("❌ No se logró pasar el CAPTCHA / obtener filas tras varios intentos.")
        return False

    # -------------------------- Extracción ---------------------------------

    async def extract_page(self, page: Page) -> List[Dict[str, Any]]:
        """Extraer las filas de la tabla de la página actual."""
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

                // Intento amplio de selección de filas
                let rows = Array.from(document.querySelectorAll('table tbody tr'));
                if (!rows.length) {
                    rows = Array.from(document.querySelectorAll('tbody tr'));
                }

                rows.forEach(row => {
                    const cells = row.querySelectorAll('td');
                    if (cells.length < 4) return;

                    const item = {
                        numero_convocatoria: clean(cells[0]?.innerText || ''),
                        unidad_organica:    clean(cells[1]?.innerText || ''),
                        descripcion:        clean(cells[2]?.innerText || ''),
                        cierre_postulacion: clean(cells[3]?.innerText || ''),
                    };

                    const links = row.querySelectorAll('a');
                    links.forEach(a => {
                        const txt = clean(a.textContent || '').toLowerCase();
                        const href = a.href || '';

                        if (href && href.toLowerCase().includes('pdf')) {
                            item.tdr_url = href;
                        } else if (txt.includes('ver')) {
                            item.detalle_url = href;
                        }
                    });

                    if (item.numero_convocatoria) {
                        item.fecha_extraccion = new Date().toISOString();
                        results.push(item);
                    }
                });

                return results;
            }
            """
        )

        print(f"✅ Filas extraídas en esta página: {len(data)}")
        return data

    async def paginate_and_extract(self, page: Page) -> List[Dict[str, Any]]:
        """Recorrer páginas usando 'Siguiente' y acumular filas."""
        all_rows: List[Dict[str, Any]] = []

        for idx in range(self.max_pages):
            print(f"📄 Página {idx + 1}/{self.max_pages}")
            rows = await self.extract_page(page)
            all_rows.extend(rows)

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

        print(f"📊 Total filas acumuladas: {len(all_rows)}")
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

                # Resolver captcha + Buscar con reintentos
                ok = await self.solve_and_search_with_retries(page)
                if not ok:
                    print("❌ No se logró obtener tabla con filas tras reintentos.")
                    return {
                        "success": False,
                        "timestamp": datetime.now().isoformat(),
                        "source_url": self.url,
                        "total_convocatorias": 0,
                        "convocatorias": [],
                        "error": "No se logró pasar CAPTCHA / obtener filas tras reintentos",
                    }

                # Extraer y paginar
                rows = await self.paginate_and_extract(page)

                result: Dict[str, Any] = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_file = f"data/raw/pj_8uit_raw_{ts}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    processed = {
        "metadata": {
            "source": "pj_8uit_convocatorias",
            "extraction_timestamp": result.get("timestamp", datetime.now().isoformat()),
            "total_records": result.get("total_convocatorias", 0),
        },
        "convocatorias": result.get("convocatorias", []),
    }
    proc_file = f"data/processed/pj_8uit_tdr_{ts}.json"
    with open(proc_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print("📁 Archivos guardados:")
    print(f"  RAW:  {raw_file}")
    print(f"  PROC: {proc_file}")

    return 0 if result.get("success") else 1


async def run_local() -> None:
    """Modo debug local (abre navegador visible)."""
    scraper = PJScraper(
        headless=False,
        timeout=90,
        captcha_code=None,  # o un código fijo para probar
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
