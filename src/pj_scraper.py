#!/usr/bin/env python3
"""
PJ 8UIT Scraper - VERSIÓN COMPLETA MEJORADA

INCLUYE:
✅ Manejo de CAPTCHA con Tesseract OCR
✅ Extracción de características técnicas de PDFs (texto + OCR fallback)
✅ Configuración robusta de browser y página
✅ Manejo de overlays, modales y cookies
✅ Paginación inteligente mejorada (nueva funcionalidad)
✅ Detección automática de total de registros
✅ Navegación con flechas y esperas de 10 segundos
✅ Manejo de errores granular por fila
✅ Todas las funciones originales optimizadas

FLUJO MEJORADO:
1. Detectar total de registros disponibles
2. Cambiar a 100 registros por página
3. Esperar 10 segundos para estabilizar
4. Procesar cada página descargando todos los TDRs
5. Navegar solo cuando sea necesario (más de 100 registros)
6. Terminar automáticamente cuando se completen todos
"""

import asyncio
import json
import os
import sys
import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo
from playwright.async_api import async_playwright, Page

from PIL import Image, ImageOps, ImageFilter
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path  # <-- para OCR en PDFs (fallback)

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
    OCR avanzado para el CAPTCHA.
    No se toca: se usa tal cual para resolver el captcha.
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

    best = ""
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


def _extract_caracteristicas_block(full_text: str) -> Optional[str]:
    """
    Lógica común para encontrar el bloque 'CARACTERISTICAS TECNICAS'
    dentro de un texto completo (ya sea extraído con PyPDF2 o por OCR).
    """
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
        return None

    # Posibles encabezados que marcan el final del bloque
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


def extract_caracteristicas_from_pdf(
    pdf_path: str,
    enable_ocr_fallback: bool = True,
) -> tuple[Optional[str], bool]:
    """
    Intenta extraer 'CARACTERISTICAS TECNICAS' de un PDF.

    Intento 1: texto embebido (PyPDF2).
    Intento 2 (fallback, si enable_ocr_fallback=True): OCR de imágenes (pdf2image + pytesseract).

    Devuelve:
      (bloque_texto_o_None, used_ocr)
    """
    text_pages: List[str] = []
    used_ocr: bool = False

    # ---------------- Intento 1: PyPDF2.extract_text ----------------
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"⚠️ No se pudo abrir PDF '{pdf_path}': {e}")
        return None, False

    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            text_pages.append(t)

    if text_pages:
        full_text = "\n".join(text_pages)
        block = _extract_caracteristicas_block(full_text)
        if block:
            print(f"✅ Bloque 'CARACTERISTICAS TECNICAS' obtenido desde texto embebido: {pdf_path}")
            return block, False

    # Si no hay texto o no se encontró el bloque → fallback OCR (segundo intento)
    if not enable_ocr_fallback:
        return None, False

    # ---------------- Intento 2: OCR sobre imágenes del PDF ----------------
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"⚠️ Fallback OCR: no se pudo rasterizar '{pdf_path}' a imágenes: {e}")
        return None, False

    used_ocr = True
    ocr_texts: List[str] = []

    for idx, img in enumerate(images):
        try:
            gray = img.convert("L")
            txt = pytesseract.image_to_string(gray)
            if txt.strip():
                ocr_texts.append(txt)
        except Exception as e:
            print(f"⚠️ Error OCR en página {idx} de '{pdf_path}': {e}")

    if not ocr_texts:
        print(f"⚠️ Fallback OCR: no se obtuvo texto OCR para '{pdf_path}'")
        return None, used_ocr

    full_ocr_text = "\n".join(ocr_texts)
    block_ocr = _extract_caracteristicas_block(full_ocr_text)
    if block_ocr:
        print(f"✅ Bloque 'CARACTERISTICAS TECNICAS' obtenido por OCR: {pdf_path}")
    else:
        print(f"ℹ️ Fallback OCR: sin bloque 'CARACTERISTICAS TECNICAS' en '{pdf_path}'")

    return block_ocr, used_ocr


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
# Clase principal del scraper MEJORADA
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

    # ================ NUEVAS FUNCIONES MEJORADAS ================

    async def close_any_modals(self, page: Page) -> None:
        """
        MEJORADO: Cierra modales SweetAlert2 y otros que interfieren con la navegación
        """
        try:
            # Lista de selectores para diferentes tipos de modales
            modal_selectors = [
                '.swal2-container',
                '.swal2-popup', 
                '.swal2-backdrop-show',
                '.modal:visible',
                '.popup:visible',
                '[role="dialog"]:visible',
                '.overlay:visible'
            ]
            
            for selector in modal_selectors:
                try:
                    modal = page.locator(selector).first
                    if await modal.is_visible(timeout=1000):
                        print(f"🔴 Cerrando modal: {selector}")
                        
                        # Intentar ESC primero
                        await page.keyboard.press("Escape")
                        await asyncio.sleep(0.3)
                        
                        # Si persiste, buscar botón de cierre
                        if await modal.is_visible(timeout=500):
                            close_selectors = [
                                '.swal2-close',
                                '.swal2-cancel', 
                                '[data-dismiss]',
                                '.close',
                                '.btn-close'
                            ]
                            
                            for close_sel in close_selectors:
                                try:
                                    close_btn = modal.locator(close_sel).first
                                    if await close_btn.is_visible(timeout=300):
                                        await close_btn.click()
                                        break
                                except:
                                    continue
                        
                        break  # Solo cerrar el primer modal encontrado
                        
                except:
                    continue
                    
            await asyncio.sleep(0.2)  # Pequeña pausa para estabilizar
            
        except Exception as e:
            # Error silencioso - no queremos que esto bloquee el flujo principal
            pass

    async def detect_total_records(self, page: Page) -> int:
        """
        NUEVO: Detecta el total de registros analizando textos de paginación
        Patrones: "1 - 20 de 62", "Mostrando 1-20 de 234", etc.
        """
        try:
            await self.close_any_modals(page)
            
            total_detected = await page.evaluate(
                """
                () => {
                    // Patrones para detectar total de registros
                    const patterns = [
                        /\\b(\\d+)\\s*-\\s*\\d+\\s*de\\s*(\\d+)\\b/i,     // "1 - 20 de 62"
                        /\\b\\d+\\s*de\\s*(\\d+)\\s*registros?\\b/i,      // "20 de 62 registros"  
                        /\\btotal:?\\s*(\\d+)\\b/i,                       // "Total: 62"
                        /\\b(\\d+)\\s*registros?\\s*encontrados?\\b/i,    // "62 registros encontrados"
                        /\\bshowing\\s*\\d+-\\d+\\s*of\\s*(\\d+)/i,       // "Showing 1-20 of 62"
                        /\\d+\\s*\\/\\s*(\\d+)/                           // "20/62"
                    ];
                    
                    // Buscar en todos los elementos visibles
                    const allElements = Array.from(document.querySelectorAll('*'));
                    const candidates = [];
                    
                    for (const el of allElements) {
                        // Solo elementos visibles y con texto relevante
                        if (el.offsetHeight > 0 && el.offsetWidth > 0) {
                            const text = (el.innerText || el.textContent || '').trim();
                            
                            if (text && text.length < 200) {  // Evitar textos muy largos
                                for (const pattern of patterns) {
                                    const match = text.match(pattern);
                                    if (match) {
                                        const total = parseInt(match[match.length - 1]);  // Último grupo capturado
                                        if (total > 0 && total <= 50000) {  // Validación razonable
                                            candidates.push({
                                                total: total,
                                                text: text.substring(0, 100),
                                                pattern: pattern.toString()
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Retornar el candidato más probable
                    if (candidates.length > 0) {
                        // Ordenar por confianza (patrón más específico primero)
                        candidates.sort((a, b) => {
                            if (a.text.includes('de') && !b.text.includes('de')) return -1;
                            if (!a.text.includes('de') && b.text.includes('de')) return 1;
                            return b.total - a.total;  // Mayor número como fallback
                        });
                        
                        console.log('Detected totals:', candidates.slice(0, 3));
                        return candidates[0].total;
                    }
                    
                    return 0;
                }
                """
            )
            
            return total_detected || 0
            
        except Exception as e:
            print(f"⚠️ Error detectando total: {e}")
            return 0

    async def select_page_size_100_FIXED(self, page: Page) -> None:
        """
        CORREGIDO: Intenta cambiar 'Registros por página' a 100 
        Basado en la interfaz real de Material Design
        """
        try:
            # Primero cerrar cualquier modal que pueda estar abierto
            await self.close_any_modals(page)
            
            print("📏 Intentando cambiar a 100 registros por página...")
            
            # Estrategia 1: Buscar el dropdown específico de "Registros por página"
            page_size_selectors = [
                # Selector específico del dropdown de Material Design
                '.mat-mdc-select[aria-label*="página"]',
                '.mat-select[aria-label*="página"]',
                # Buscar por texto "Registros por página"
                'mat-select:near(:text("Registros por página"))',
                # Selectores más genéricos
                '.mat-mdc-select',
                '.mat-select',
                'select',
            ]
            
            dropdown_clicked = False
            for sel in page_size_selectors:
                try:
                    print(f"🔍 Intentando dropdown selector: {sel}")
                    dropdown = page.locator(sel).first
                    if await dropdown.is_visible(timeout=3000):
                        print(f"✅ Encontrado dropdown: {sel}")
                        await dropdown.scroll_into_view_if_needed()
                        await asyncio.sleep(1)
                        await dropdown.click()
                        dropdown_clicked = True
                        break
                except Exception as e:
                    print(f"⚠️ Error con selector {sel}: {str(e)[:50]}...")
                    continue
            
            if dropdown_clicked:
                print("✅ Dropdown abierto, buscando opción '100'...")
                await asyncio.sleep(2)  # Esperar a que aparezcan las opciones
                
                # Buscar la opción "100" en el dropdown abierto
                option_selectors = [
                    'mat-option:has-text("100")',
                    '.mat-option:has-text("100")',
                    '.mat-mdc-option:has-text("100")',
                    'div[role="option"]:has-text("100")',
                    '[role="option"]:has-text("100")',
                ]
                
                option_clicked = False
                for opt_sel in option_selectors:
                    try:
                        option = page.locator(opt_sel).first
                        if await option.is_visible(timeout=3000):
                            print(f"✅ Encontrada opción 100: {opt_sel}")
                            await option.click()
                            option_clicked = True
                            break
                    except Exception:
                        continue
                
                if not option_clicked:
                    print("⚠️ No se encontró opción '100', intentando JavaScript...")
                    await page.evaluate(
                        """
                        () => {
                            // Buscar opciones con texto "100"
                            const options = Array.from(document.querySelectorAll('[role="option"], .mat-option, .mat-mdc-option'));
                            for (const opt of options) {
                                if ((opt.innerText || opt.textContent || '').trim() === '100') {
                                    opt.click();
                                    return true;
                                }
                            }
                            return false;
                        }
                        """
                    )
            
            # Estrategia 2: JavaScript directo si la estrategia 1 falla
            if not dropdown_clicked:
                print("🔍 Intentando estrategia JavaScript para dropdown...")
                success = await page.evaluate(
                    """
                    () => {
                        // Buscar dropdown de registros por página
                        const dropdowns = Array.from(document.querySelectorAll('.mat-select, .mat-mdc-select, select'));
                        
                        for (const dropdown of dropdowns) {
                            const parent = dropdown.closest('[class*="paginator"], [class*="pagination"]') || 
                                         dropdown.parentElement;
                            const parentText = (parent?.innerText || '').toLowerCase();
                            
                            if (parentText.includes('registros') && parentText.includes('página')) {
                                console.log('Found page size dropdown:', dropdown);
                                dropdown.click();
                                
                                // Esperar un poco y buscar la opción 100
                                setTimeout(() => {
                                    const options = Array.from(document.querySelectorAll('[role="option"], .mat-option, .mat-mdc-option'));
                                    for (const opt of options) {
                                        if ((opt.innerText || opt.textContent || '').trim() === '100') {
                                            opt.click();
                                            return;
                                        }
                                    }
                                }, 1000);
                                
                                return true;
                            }
                        }
                        return false;
                    }
                    """
                )
                
                if success:
                    print("✅ Dropdown encontrado vía JavaScript")
            
            # Esperar a que la tabla se recargue con 100 registros
            print("⏱️ Esperando a que la tabla se recargue con 100 registros...")
            await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
            await asyncio.sleep(3)  # Espera adicional para estabilizar
            await self.close_any_modals(page)  # Cerrar modales post-operación
            
            print("✅ Intento de establecer 100 registros por página completado.")
            
        except Exception as e:
            print(f"⚠️ No se pudo ajustar 'Registros por página' a 100: {e}")
            print("ℹ️ Continuando con configuración actual...")

    async def try_next_page_robust(self, page: Page) -> bool:
        """
        MEJORADO: Navega a la siguiente página usando flechas de Material Design
        Incluye espera de 10 segundos como especifica Oscar
        """
        # Selectores específicos para flechas de "siguiente página"
        arrow_selectors = [
            '.mat-mdc-paginator-navigation-next:not([disabled]):not([aria-disabled="true"])',
            'button[aria-label*="siguiente"]:not([disabled]):not([aria-disabled="true"])',
            'button[aria-label*="next"]:not([disabled]):not([aria-disabled="true"])',
            'button[class*="next"]:not([disabled]):not([aria-disabled="true"])',
            'button[class*="paginator"]:not([disabled]):not([aria-disabled="true"])'
        ]
        
        print("🔍 Buscando flecha 'siguiente página'...")
        
        # Estrategia 1: Locators de Playwright
        for selector in arrow_selectors:
            try:
                await self.close_any_modals(page)
                
                arrow_btn = page.locator(selector).first
                if await arrow_btn.is_visible(timeout=3000):
                    # Verificar que no esté deshabilitado
                    disabled = await arrow_btn.get_attribute("disabled")
                    aria_disabled = await arrow_btn.get_attribute("aria-disabled")
                    
                    if disabled is None and aria_disabled != "true":
                        print(f"➡️ Encontrada flecha activa: {selector}")
                        
                        await arrow_btn.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)
                        await arrow_btn.click()
                        
                        print("⏱️ ESPERANDO 10 SEGUNDOS (navegación)...")
                        await asyncio.sleep(10)
                        return True
                    else:
                        print(f"⚠️ Flecha deshabilitada: {selector}")
                
            except Exception as e:
                print(f"⚠️ Error con selector {selector}: {str(e)[:40]}...")
                continue
        
        # Estrategia 2: JavaScript como fallback
        try:
            print("🔍 Intentando JavaScript para navegación...")
            await self.close_any_modals(page)
            
            clicked = await page.evaluate(
                """
                () => {
                    // Limpiar modales SweetAlert2
                    document.querySelectorAll('.swal2-container').forEach(el => el.remove());
                    
                    // Buscar botones de navegación
                    const buttons = Array.from(document.querySelectorAll('button'));
                    
                    for (const btn of buttons) {
                        const classes = btn.className || '';
                        const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                        
                        if ((classes.includes('mat-mdc-paginator-navigation-next') ||
                             classes.includes('navigation-next') ||
                             ariaLabel.includes('siguiente') ||
                             ariaLabel.includes('next')) &&
                            !btn.disabled && 
                            btn.getAttribute('aria-disabled') !== 'true') {
                            
                            console.log('Clicking navigation button:', btn.className);
                            btn.click();
                            return true;
                        }
                    }
                    
                    return false;
                }
                """
            )
            
            if clicked:
                print("✅ Navegación exitosa vía JavaScript")
                print("⏱️ ESPERANDO 10 SEGUNDOS (navegación JS)...")
                await asyncio.sleep(10)
                return True
            
        except Exception as e:
            print(f"⚠️ Error en navegación JavaScript: {e}")
        
        print("❌ No se encontró flecha de navegación activa")
        return False

    # ============= PAGINACIÓN INTELIGENTE MEJORADA =============

    async def paginate_and_extract_INTELLIGENT(self, page: Page) -> List[Dict[str, Any]]:
        """
        LÓGICA INTELIGENTE según Oscar:
        1. Detectar total de registros disponibles
        2. Cambiar a 100 registros por página  
        3. Esperar 10 segundos para estabilizar
        4. Procesar cada página descargando todos los TDRs
        5. Si hay más páginas (total > registros procesados), navegar y repetir
        6. Terminar cuando se hayan procesado todos los registros
        """
        print("🧠 === INICIANDO PAGINACIÓN INTELIGENTE ===")
        
        # PASO 1: DETECTAR TOTAL DE REGISTROS DISPONIBLES
        print("📊 PASO 1: Detectando total de registros...")
        total_registros = await self.detect_total_records(page)
        
        if total_registros == 0:
            print("❌ No se detectaron registros. Asumiendo procesamiento manual.")
            total_registros = 999  # Procesar hasta encontrar fin natural
        else:
            print(f"✅ TOTAL DETECTADO: {total_registros} registros")
        
        # PASO 2: CAMBIAR A 100 REGISTROS POR PÁGINA
        print("📏 PASO 2: Cambiando a 100 registros por página...")
        await self.close_any_modals(page)
        await self.select_page_size_100_FIXED(page)
        
        # PASO 3: ESPERAR 10 SEGUNDOS
        print("⏱️ PASO 3: Esperando 10 segundos para estabilizar...")
        await asyncio.sleep(10)
        await self.close_any_modals(page)
        
        # CALCULAR PÁGINAS NECESARIAS
        if total_registros == 999:
            paginas_estimadas = "desconocido"
            max_paginas = self.max_pages  # Usar límite por defecto
        else:
            max_paginas = max(1, (total_registros + 99) // 100)  # Redondear hacia arriba
            paginas_estimadas = max_paginas
            
        print(f"📄 PÁGINAS ESTIMADAS: {paginas_estimadas}")
        
        all_rows: List[Dict[str, Any]] = []
        pagina_actual = 1
        
        # PASO 4: PROCESAR PÁGINAS SECUENCIALMENTE
        while pagina_actual <= max_paginas:
            print(f"\n🔄 === PROCESANDO PÁGINA {pagina_actual} ===")
            
            # Limpiar modales antes de procesar
            await self.close_any_modals(page)
            
            # Verificar estado actual de la página
            page_info = await self.verify_current_page_info(page, pagina_actual)
            
            # Extraer filas de la página actual
            page_rows = await self.extract_page(page)
            
            if not page_rows:
                print("⚠️ No hay filas en esta página.")
                if pagina_actual == 1:
                    print("❌ ERROR: Primera página vacía. Terminando.")
                    break
                else:
                    print("✅ Fin natural de datos. Terminando procesamiento.")
                    break
            
            print(f"📋 FILAS ENCONTRADAS: {len(page_rows)}")
            
            # Procesar cada fila de la página (descargar TDRs)
            successful_downloads = await self.process_all_page_rows(page, page_rows)
            
            # Agregar filas procesadas al resultado final
            for item in page_rows:
                item.pop("_row_index_in_page", None)
                all_rows.append(item)
            
            print(f"✅ PÁGINA {pagina_actual} COMPLETADA: {len(page_rows)} registros | {successful_downloads} TDRs descargados")
            
            # PASO 5: DECIDIR SI CONTINUAR A SIGUIENTE PÁGINA
            registros_procesados = len(all_rows)
            
            # Condición de parada mejorada
            if total_registros != 999 and registros_procesados >= total_registros:
                print(f"🎯 TODOS LOS REGISTROS PROCESADOS: {registros_procesados}/{total_registros}")
                break
            
            if pagina_actual >= max_paginas:
                print("🎯 LÍMITE DE PÁGINAS ALCANZADO")
                break
                
            # Si hay más registros por procesar, navegar a siguiente página
            print(f"➡️ HAY MÁS REGISTROS. Navegando a página {pagina_actual + 1}...")
            
            await self.close_any_modals(page)
            next_success = await self.try_next_page_robust(page)
            
            if not next_success:
                print("❌ No se pudo navegar a la siguiente página. Finalizando.")
                break
            
            print("✅ Navegación exitosa. Esperando carga...")
            
            # Esperar carga de la nueva página
            try:
                await page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                await asyncio.sleep(2)  # Estabilización adicional
            except Exception as e:
                print(f"⚠️ Warning durante carga: {e}")
                
            pagina_actual += 1
        
        print(f"\n🎉 === PROCESAMIENTO COMPLETADO ===")
        print(f"📊 RESULTADO FINAL:")
        print(f"   • Registros esperados: {total_registros}")
        print(f"   • Registros obtenidos: {len(all_rows)}")
        print(f"   • Páginas procesadas: {pagina_actual}")
        
        return all_rows

    async def verify_current_page_info(self, page: Page, expected_page: int) -> dict:
        """
        NUEVO: Verifica qué registros están siendo mostrados en la página actual
        """
        try:
            page_info = await page.evaluate(
                """
                () => {
                    const allElements = Array.from(document.querySelectorAll('*'));
                    
                    for (const el of allElements) {
                        if (el.offsetHeight > 0 && el.offsetWidth > 0) {
                            const text = (el.innerText || '').trim();
                            
                            // Buscar patrón "X - Y de Z"
                            const match = text.match(/(\\d+)\\s*-\\s*(\\d+)\\s*de\\s*(\\d+)/i);
                            if (match) {
                                return {
                                    desde: parseInt(match[1]),
                                    hasta: parseInt(match[2]), 
                                    total: parseInt(match[3]),
                                    text: text.substring(0, 100)
                                };
                            }
                        }
                    }
                    
                    return null;
                }
                """
            )
            
            if page_info:
                print(f"📄 Página {expected_page}: {page_info['desde']}-{page_info['hasta']} de {page_info['total']}")
                return page_info
            else:
                print(f"📄 Página {expected_page}: Info de rango no disponible")
                return {}
                
        except Exception as e:
            print(f"⚠️ Error verificando página: {e}")
            return {}

    async def process_all_page_rows(self, page: Page, page_rows: List[Dict[str, Any]]) -> int:
        """
        NUEVO: Procesa todas las filas de una página descargando TDRs
        Retorna cantidad de descargas exitosas
        """
        if not page_rows:
            return 0
        
        print(f"📥 Procesando {len(page_rows)} filas de la página...")
        
        # Obtener locators de filas
        rows_locator = page.locator("table tbody tr")
        cnt = await rows_locator.count()
        if cnt == 0:
            rows_locator = page.locator("tbody tr")
            cnt = await rows_locator.count()
        
        successful_downloads = 0
        
        for idx, item in enumerate(page_rows):
            row_idx = item.get("_row_index_in_page")
            if row_idx is None or row_idx >= cnt:
                continue
            
            row_loc = rows_locator.nth(row_idx)
            numero = item.get('numero_convocatoria', f'Fila-{idx+1}')
            
            print(f"  📄 {idx+1}/{len(page_rows)}: {numero}")
            
            # Cerrar modales antes de cada operación
            await self.close_any_modals(page)
            
            # Intentar descarga con manejo de errores
            try:
                await self.enrich_row_with_tdr_pdf_IMPROVED(page, row_loc, item)
                
                if item.get("tdr_downloaded"):
                    successful_downloads += 1
                    print(f"    ✅ TDR descargado: {item.get('tdr_filename', 'N/A')}")
                else:
                    print(f"    ⚠️ TDR no disponible")
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)[:60]}...")
                # Marcar como fallo pero continuar
                item["tdr_downloaded"] = False
                item["tdr_filename"] = None
                item["caracteristicas_tecnicas"] = None
                item["caracteristicas_tecnicas_ocr"] = False
        
        print(f"  📊 Resumen: {successful_downloads}/{len(page_rows)} TDRs exitosos")
        return successful_downloads

    # ======================== FUNCIONES ORIGINALES OPTIMIZADAS ========================

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
        Añade _row_index_in_page para luego localizar la fila real.
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

    async def enrich_row_with_tdr_pdf_IMPROVED(
        self,
        page: Page,
        row_locator,
        item: Dict[str, Any],
    ) -> None:
        """
        MEJORADO: Con timeout más corto y mejor manejo de errores
        Para una fila concreta:
        - Intenta localizar el elemento clicable del TDR.
        - Usa expect_download() para capturar el PDF.
        - Extrae 'caracteristicas_tecnicas' del PDF (texto -> OCR fallback).
        """
        try:
            # Cerrar modales antes de intentar click
            await self.close_any_modals(page)
            
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
            
            # TIMEOUT REDUCIDO PARA EVITAR BLOQUEOS
            try:
                async with page.expect_download(timeout=30000) as dl_info:  # 30s en lugar de 90s
                    await candidate.click(timeout=10000)  # 10s timeout para el click
                download = await dl_info.value
            except Exception as e:
                print(f"⚠️ No se produjo descarga para la fila (timeout reducido): {str(e)[:50]}...")
                return

            tmp_path = await download.path()
            if not tmp_path:
                print("⚠️ Descarga sin ruta de archivo (tmp_path vacío).")
                return

            suggested_name = download.suggested_filename or os.path.basename(tmp_path)
            item["tdr_filename"] = suggested_name
            item["tdr_downloaded"] = True

            # Extraer bloque de CARACTERISTICAS TECNICAS
            block, used_ocr = extract_caracteristicas_from_pdf(tmp_path, enable_ocr_fallback=True)
            item["caracteristicas_tecnicas"] = block
            item["caracteristicas_tecnicas_ocr"] = bool(used_ocr)

        except Exception as e:
            print(f"⚠️ Error enriqueciendo fila con TDR (MEJORADO): {str(e)[:100]}...")

    # -------------------------- Ejecución ----------------------------------

    async def run(self) -> Dict[str, Any]:
        print("🚀 Iniciando scraper PJ 8UIT MEJORADO...")
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

                # ============ USAR PAGINACIÓN INTELIGENTE ============
                rows_raw = await self.paginate_and_extract_INTELLIGENT(page)
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
