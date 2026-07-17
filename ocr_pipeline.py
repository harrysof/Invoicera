"""
ocr_pipeline.py

Core pipeline: PDF -> page images -> PP-StructureV3 -> structured JSON/Markdown.

This module has NO GUI code in it on purpose. It should be usable from
a script, a notebook, or the GUI equally. Keep it that way.

ENGINE: this uses PaddleOCR 3.x's PPStructureV3 pipeline (from the `paddleocr`
pip package with the `[doc-parser]` extras group), NOT PaddleOCR-json.

Why the switch back from PaddleOCR-json: PP-StructureV3 does real layout/table
structure recognition (reading order, table detection, row/column structure)
built into the model itself, instead of the row-position heuristic we wrote
by hand for PaddleOCR-json's plain-text output. PaddleOCR-json's underlying
engine is frozen on PP-OCR v2.x/v3/v4 with no structure understanding at all
-- fine for plain text, not for invoices where you actually want the table
shape.

VERSION PINS (see requirements.txt) -- do not casually bump these:
    paddlepaddle == 3.0.0
    paddleocr[doc-parser] == 3.7.0

This combination was verified here: installs cleanly with `pip check`
reporting no conflicts, and PPStructureV3 imports and constructs correctly
(confirmed up to the point of downloading model weights, which requires
network access this sandbox doesn't have). paddlepaddle==3.0.0 specifically
matches the version PaddleOCR's own team used in their tested PyInstaller
packaging recipe (see README) -- don't bump it without re-checking that
packaging still works, since that's the whole reason for pinning to their
tested matrix rather than "latest".
"""

import json
from pathlib import Path

# Import is deferred into get_pipeline() rather than done at module load time.
# Constructing PPStructureV3 loads/validates model config immediately, which
# is slow and which we don't want to pay just for importing this module
# (e.g. if some other part of the app imports ocr_pipeline without OCR-ing
# anything yet).
_pipeline = None


def get_pipeline(text_recognition_model_name="en_PP-OCRv5_mobile_rec"):
    """
    Get (constructing if needed) the single shared PPStructureV3 pipeline.
    Reused across pages/files -- constructing it loads several models and is slow.

    Table recognition is ON (the whole point for invoices). Formula, chart,
    and seal recognition are OFF -- invoices essentially never contain any
    of those, and disabling them means fewer models to download and faster
    inference. Flip them on here if you ever need them.

    text_recognition_model_name defaults to "en_PP-OCRv5_mobile_rec" instead
    of the pipeline's own default ("PP-OCRv5_server_rec", a multilingual
    model). Reason: the multilingual default was observed dropping spaces
    between words on a real invoice (e.g. "Désignationduredevable" instead
    of "Désignation du redevable") -- a long-documented PP-OCR behavior
    (see PaddlePaddle/PaddleOCR#5448 from 2022, still relevant today).
    PaddleOCR's own model card for en_PP-OCRv5_mobile_rec describes it as
    fine-tuned specifically for English and explicitly claims better space
    handling than the general multilingual model. Since these invoices are
    French/English/Latin-script (not Chinese/Japanese, which the default
    model is also tuned for), the English-specific model is a better fit
    and should help with the missing-space issue -- untested on your actual
    documents yet, since this sandbox can't download model weights. If your
    invoices lean more French, "latin_PP-OCRv5_mobile_rec" is the other
    documented option worth trying if English-specific still misses spaces
    on French words.
    """
    global _pipeline
    if _pipeline is None:
        from paddleocr import PPStructureV3
        _pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_table_recognition=True,
            text_recognition_model_name=text_recognition_model_name,
        )
    return _pipeline


def pdf_to_images(pdf_path, dpi=200):
    """
    Render each page of a PDF to a temp PNG file on disk.
    PPStructureV3's .predict() accepts a file path directly.

    Returns (list of image file paths, temp directory to clean up after).
    """
    import fitz  # PyMuPDF
    import tempfile

    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # PDF points are 72 dpi by default
    mat = fitz.Matrix(zoom, zoom)

    out_dir = Path(tempfile.mkdtemp(prefix="invoicera_ocr_"))

    image_paths = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat)
        out_path = out_dir / f"page_{i:03d}.png"
        pix.save(str(out_path))
        image_paths.append(out_path)
    doc.close()
    return image_paths, out_dir


def _group_boxes_into_rows(boxes, y_tolerance_ratio=0.5):
    """
    Group cell boxes (each [left, top, right, bottom]) into visual rows by
    vertical position, sorted left-to-right within each row. Same idea as
    the row-grouping heuristic from the old PaddleOCR-json pipeline, but
    applied here to the table model's own detected cell boxes -- not raw
    OCR text -- purely as an independent check on the HTML's row/column
    counts, never as the primary source of table content.
    """
    if not boxes:
        return []

    items = [
        {"box": b, "y_center": (b[1] + b[3]) / 2, "x_center": (b[0] + b[2]) / 2, "height": b[3] - b[1]}
        for b in boxes
    ]
    items.sort(key=lambda i: i["y_center"])

    avg_height = sum(i["height"] for i in items) / len(items) or 10
    tolerance = avg_height * y_tolerance_ratio

    rows = []
    current_row = [items[0]]
    current_y = items[0]["y_center"]
    for item in items[1:]:
        if abs(item["y_center"] - current_y) <= tolerance:
            current_row.append(item)
            current_y = sum(i["y_center"] for i in current_row) / len(current_row)
        else:
            current_row.sort(key=lambda i: i["x_center"])
            rows.append(current_row)
            current_row = [item]
            current_y = item["y_center"]
    current_row.sort(key=lambda i: i["x_center"])
    rows.append(current_row)
    return rows


def _html_row_cell_counts(pred_html):
    """
    Parse an HTML table string and return the number of <td>/<th> cells in
    each <tr>. Uses Python's built-in html.parser rather than adding a
    dependency like BeautifulSoup, since this only needs cell counts per
    row, not full DOM traversal.
    """
    from html.parser import HTMLParser

    class RowCounter(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self._current_row_cells = None

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self._current_row_cells = 0
            elif tag in ("td", "th") and self._current_row_cells is not None:
                self._current_row_cells += 1

        def handle_endtag(self, tag):
            if tag == "tr" and self._current_row_cells is not None:
                self.rows.append(self._current_row_cells)
                self._current_row_cells = None

    parser = RowCounter()
    parser.feed(pred_html)
    return parser.rows


def check_table_structure(table_res):
    """
    Cross-check a single table's HTML structure against its independently-
    detected cell boxes. These come from two different models in the
    pipeline (structure prediction generates the HTML's <td> sequence;
    cell detection finds cell boundaries visually) -- when they disagree
    on how many cells are in a row, that's a real signal something went
    wrong, not a heuristic guess.

    Known failure mode this catches: PP-StructureV3's table structure
    model can under-predict <td> tags for rows containing empty cells
    (see PaddlePaddle/PaddleOCR#8807 -- empty cells aren't well-represented
    in the model's training data). When that happens, every cell after the
    dropped one shifts left by one column in the HTML, silently corrupting
    the row -- e.g. a REFERENCE value ending up under the DESIGNATION
    header. The HTML stays well-formed, so nothing else would catch this.

    Returns:
        {
            "ok": bool,               # True if HTML and detected cell rows agree
            "html_row_counts": [...], # cells per row per the HTML
            "detected_row_counts": [...],  # cells per row per cell_box_list
            "mismatch_rows": [...],   # indices where they disagree
        }

    This is NOT a fix -- it doesn't know which count is right, or how to
    correct the HTML. It's a flag: "this table's structure may be wrong,
    don't trust it blindly." What to do with that flag (reject, re-OCR,
    route to a human, warn the downstream LLM) is a decision for whoever
    consumes this pipeline's output, not something to silently resolve here.
    """
    pred_html = table_res.get("pred_html", "")
    cell_box_list = table_res.get("cell_box_list", [])

    html_row_counts = _html_row_cell_counts(pred_html)
    detected_rows = _group_boxes_into_rows(cell_box_list)
    detected_row_counts = [len(row) for row in detected_rows]

    mismatch_rows = []
    for i in range(min(len(html_row_counts), len(detected_row_counts))):
        if html_row_counts[i] != detected_row_counts[i]:
            mismatch_rows.append(i)
    # Also flag if the two methods found a different number of rows entirely.
    if len(html_row_counts) != len(detected_row_counts):
        mismatch_rows.append("row_count_mismatch")

    return {
        "ok": len(mismatch_rows) == 0,
        "html_row_counts": html_row_counts,
        "detected_row_counts": detected_row_counts,
        "mismatch_rows": mismatch_rows,
    }


def process_page(image_path):
    """
    Run PP-StructureV3 on a single page image.

    Returns a dict:
        {
            "markdown": str,               # reading-order-aware markdown for this page
            "json": dict,                  # the pipeline's own structured JSON result
            "table_warnings": [...],       # per-table structure-check results (see check_table_structure)
        }

    PPStructureV3.predict() returns an iterable of result objects (one per
    input image/page). Each result object has .markdown (a dict with
    'markdown_texts' among other keys) and can be serialized via
    save_to_json()/save_to_markdown(), or accessed as a dict directly.
    """
    pipeline = get_pipeline()
    results = list(pipeline.predict(str(image_path)))

    if not results:
        return {"markdown": "", "json": {}, "table_warnings": []}

    res = results[0]

    # res.markdown is a dict: {"markdown_texts": ..., "markdown_images": ..., ...}
    # (MARKDOWN_SAVE_KEYS confirms "markdown_texts" is the real key -- verified
    # directly against paddlex's MarkdownMixin source, not guessed from docs.)
    markdown_data = res.markdown
    markdown_text = markdown_data.get("markdown_texts", "") if markdown_data else ""

    # res.json is a property (JsonMixin), always present, returns {"res": {...}}.
    # Verified directly against paddlex's JsonMixin source.
    json_data = res.json

    # Cross-check every table's HTML structure against its detected cell
    # boxes. table_res_list only exists in the result when the page actually
    # has tables (use_table_recognition=True doesn't guarantee every page
    # has one), so this is empty for text-only pages -- normal, not an error.
    table_warnings = []
    table_res_list = json_data.get("res", {}).get("table_res_list", [])
    for i, table_res in enumerate(table_res_list):
        check = check_table_structure(table_res)
        if not check["ok"]:
            table_warnings.append({"table_index": i, **check})

    return {
        "markdown": markdown_text,
        "json": json_data,
        "table_warnings": table_warnings,
    }


def process_pdf(pdf_path, dpi=200):
    """
    Full pipeline for a single PDF: render pages, run PP-StructureV3 on each,
    collect per-page markdown + structured JSON, clean up temp images.

    Returns:
        {
            "file": str,
            "pages": [ {"page": 1, "markdown": str, "json": {...}}, ... ]
        }
    """
    import shutil

    pdf_path = Path(pdf_path)
    image_paths, temp_dir = pdf_to_images(pdf_path, dpi=dpi)

    try:
        pages = []
        for page_num, image_path in enumerate(image_paths, start=1):
            page_result = process_page(image_path)
            pages.append({
                "page": page_num,
                "markdown": page_result["markdown"],
                "json": page_result["json"],
                "table_warnings": page_result["table_warnings"],
            })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "file": pdf_path.name,
        "pages": pages,
    }


def combined_markdown(result):
    """
    Convenience: join every page's markdown into one document, separated by
    page-break markers. Handy for a quick look at the whole PDF's extracted
    structure at once.
    """
    parts = []
    for page in result["pages"]:
        parts.append(f"<!-- page {page['page']} -->\n{page['markdown']}")
    return "\n\n---\n\n".join(parts)
