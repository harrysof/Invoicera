# Invoicera Pipeline - PP-StructureV3 edition

This replaces the PaddleOCR-json version. Switched back to the `paddleocr` pip package
specifically for **PP-StructureV3** -- real table/layout structure recognition built into
the model, instead of the row-position heuristic we wrote by hand for plain-text OCR output.

## Why this is different from the last paddleocr3 attempt

The earlier attempt broke on unpinned/loosely-pinned versions (a 2.x/3.x API rewrite, then a
3.3.0 oneDNN CPU regression, then a numpy/paddlex conflict from the fix itself). This time:

1. **Versions are pinned to a combination verified to install cleanly together**, tested in a
   fresh venv here: `pip check` reports no broken requirements for the full stack
   (`paddlepaddle==3.0.0`, `paddleocr[doc-parser]==3.7.0`, the `paddlex==3.7.2` it resolves to,
   and `pyinstaller==6.14.2`).
2. **`paddlepaddle==3.0.0` specifically** matches the version PaddleOCR's own team used in
   their official, documented PyInstaller packaging guide -- not "whatever's newest today,"
   which is what caused churn before. Don't bump this without re-checking their packaging docs
   still apply.
3. **The `[doc-parser]` extras group is required**, not optional -- I confirmed a bare
   `paddleocr + paddlex` install (no extras) is missing `langchain`, which `paddlex` needs
   internally for an unrelated component, and crashes on import. `[doc-parser]` pulls in
   everything needed; this was tested here, not assumed.

## What I verified here, and what I couldn't

**Verified directly** (installed the real packages in a fresh venv and ran real code against them):
- The full pinned dependency stack installs with zero conflicts (`pip check` clean).
- `from paddleocr import PPStructureV3` imports successfully.
- `PPStructureV3(...)` constructs correctly with the exact toggles used in `ocr_pipeline.py`
  (`use_table_recognition=True`, formula/chart/seal recognition off).
- `.predict()` runs and gets as far as trying to download the first model's weights --
  confirming the call path, argument handling, and pipeline construction are all correct.
- The `.markdown` and `.json` properties on the result object are real (checked against
  paddlex's own source code, `MarkdownMixin`/`JsonMixin` in
  `paddlex/inference/common/result/mixin.py`), including the exact key
  (`markdown_texts`) my code reads.
- `package.py`'s dependency-detection logic runs successfully against the real installed
  environment and produces a sensible list of packages to bundle.

**Not verified** (blocked by this sandbox being Linux with no access to model-hosting servers):
- An actual OCR run producing real output -- blocked at model weight download
  (`Exception: No available model hosting platforms detected`). This will work differently
  on your machine once it can reach HuggingFace/ModelScope/AIStudio/BOS.
- `gui.py`'s Tkinter widgets -- this sandbox's Python has no `tkinter` module at all
  (unrelated to paddleocr; just not installed in this container). Same caveat as previous
  versions of this file.
- The actual `.exe` build via `package.py` -- PyInstaller building a Windows executable
  requires running on Windows. I confirmed the script's own logic works, not the final binary.

So: meaningfully more verified than last time, but the first real end-to-end run (OCR + GUI +
packaging) is still on you. If something breaks, paste the error and we'll fix it against the
real trace, same process as before.

## Update: fixed missing spaces in recognized text

First real test on an actual invoice worked well overall -- the table came through as a proper
structured HTML table (exactly the win PP-StructureV3 was supposed to deliver over the old
row-heuristic approach), and all the key fields (invoice number, date, vendor, amounts) were
captured correctly. One real issue: words were getting glued together with no spaces
(`Désignationduredevable` instead of `Désignation du redevable`).

This turned out to be a long-documented PP-OCR behavior, not a bug in this pipeline --
see PaddlePaddle/PaddleOCR#5448, a "Missing white spaces" issue open since 2022. The
pipeline's default text recognition model (`PP-OCRv5_server_rec`) is a general multilingual
model; PaddleOCR also ships `en_PP-OCRv5_mobile_rec`, described in its own model card as
fine-tuned specifically for English with better space handling.

`ocr_pipeline.py` now defaults `get_pipeline()` to `text_recognition_model_name="en_PP-OCRv5_mobile_rec"`
instead of the pipeline's own default. I confirmed this parameter is real and accepted by the
installed `PPStructureV3` constructor, and that construction still proceeds normally with it
set (gets to the same network wall as before, just further down the model-loading sequence --
no new errors introduced). What I could not verify here: whether it actually fixes the missing
spaces on your real documents, since that needs the model weights downloaded and a real OCR
run, both blocked in this sandbox.

If `en_PP-OCRv5_mobile_rec` still drops spaces on French text specifically (these invoices mix
French and English), the next thing worth trying is `text_recognition_model_name="latin_PP-OCRv5_mobile_rec"`
-- also a real documented model, tuned for the broader Latin-script language family rather than
English specifically. Change the default in `get_pipeline()`'s signature to test it.

## Setup

```bash
pip install -r requirements.txt
```

First run of `gui.py` will download PP-StructureV3's models (layout detection, OCR, table
structure recognition -- formula/chart/seal are disabled by default in `ocr_pipeline.py` since
invoices essentially never need them, which also means fewer models to download). This is a
bigger download than plain OCR was -- expect it to take a few minutes depending on your
connection.

```bash
python gui.py
```

## What changed in the output

Instead of row-grouped JSON, you now get:
- **Markdown** with real tables (`| Item | Qty | Price |` style), reading-order-aware text,
  because PP-StructureV3's layout model actually understands document structure -- not our
  hand-rolled y-position heuristic.
- **JSON** with the pipeline's full structured result (layout boxes, table structure, OCR
  text with positions, etc.) -- much richer than the old `{"row_text": ...}` shape, but also
  contains numpy arrays that get stringified (`default=str`) when saved to JSON. If you need
  those as real numbers later (e.g. precise bounding boxes), that's worth a proper serializer,
  flagged here rather than silently handled.

## Packaging as a single .exe

```bash
python package.py --file gui.py
```

This is PaddleOCR's own documented recipe (see `package.py`'s docstring for the source URL),
adapted only to default to `gui.py`. Run it from the same environment where you installed
`requirements.txt` -- it inspects your installed packages to decide what metadata to bundle.

**Important constraint from their docs**: PyInstaller is the only supported path. Their docs
explicitly state Nuitka is incompatible with PaddleOCR's packaging needs -- don't try switching
to it later for a smaller build.

The `.exe` and its dependencies land in the `dist/` folder PyInstaller creates.

**Known issues from their docs** (not something I hit, just documented by them):
- `RuntimeError: xxx requires additional dependencies` when running the built exe means the
  packaging environment was missing something `pip install -r requirements.txt` should have
  covered -- re-check that install.
- Missing CUDA/cuDNN DLL errors mean you need `--nvidia` on the packaging command, or you're
  not using GPU inference at all (in which case ignore it).

## Reverting

The PaddleOCR-json version (previous approach: row-grouping heuristic over plain OCR text,
already-compiled binary, no model downloads needed) is still available if PP-StructureV3 turns
out to be more setup than it's worth for your invoices specifically. Worth trying this version
first on a handful of real invoices before deciding either way.
