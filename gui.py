"""
gui.py

Minimal Tkinter GUI for the OCR pipeline (PP-StructureV3 backend).

Flow:
  1. User picks a single PDF or a folder of PDFs.
  2. Each PDF is processed through ocr_pipeline.process_pdf().
  3. Results are shown in a text panel as reading-order-aware Markdown
     (tables render as real Markdown tables, since PP-StructureV3 does
     genuine table structure recognition, not just text position).
  4. User can save each result as .json (full structured result) or
     .md (Markdown) into an output folder.

Threading: OCR is slow (model inference per page), so it runs on a
background thread to keep the GUI responsive. GUI updates from that
thread are marshalled back via root.after().
"""

import json
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ocr_pipeline import process_pdf, combined_markdown
from extract_fields import extract
from export_excel import build_workbook


class InvoiceraOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Invoicera - OCR Pipeline (PP-StructureV3)")
        self.root.geometry("900x650")

        self.selected_paths = []   # list of PDF file paths queued for processing
        self.results = {}          # filename -> pipeline result dict (OCR stage)
        self.extraction_results = {}  # filename -> extract_fields.extract() result
        self.output_dir = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Button(top_frame, text="Select PDF(s)", command=self.select_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Select Folder", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Run OCR", command=self.run_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Set Output Folder", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(top_frame, text="No files selected.")
        self.status_label.pack(side=tk.LEFT, padx=15)

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Split view: left = file list, right = text output for selected file
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = ttk.Frame(main_frame, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_frame, text="Files").pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(left_frame, width=35)
        self.file_listbox.pack(fill=tk.Y, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        ttk.Label(right_frame, text="Extracted Structure (Markdown)").pack(anchor=tk.W)
        self.text_output = tk.Text(right_frame, wrap=tk.WORD)
        self.text_output.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X)

        ttk.Button(bottom_frame, text="Save Selected as .json", command=lambda: self.save_selected("json")).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Save Selected as .md", command=lambda: self.save_selected("md")).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Save All as .json", command=lambda: self.save_all("json")).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Save All as .md", command=lambda: self.save_all("md")).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Extract Fields -> Excel", command=self.run_extraction).pack(side=tk.LEFT, padx=(20, 5))

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------
    def select_files(self):
        paths = filedialog.askopenfilenames(
            title="Select PDF file(s)",
            filetypes=[("PDF files", "*.pdf")],
        )
        if paths:
            self.selected_paths = list(paths)
            self._refresh_file_list()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder of PDFs")
        if folder:
            pdfs = sorted(Path(folder).glob("*.pdf"))
            if not pdfs:
                messagebox.showwarning("No PDFs found", f"No .pdf files found in {folder}")
                return
            self.selected_paths = [str(p) for p in pdfs]
            self._refresh_file_list()

    def select_output_dir(self):
        folder = filedialog.askdirectory(title="Select output folder for saved results")
        if folder:
            self.output_dir = Path(folder)
            self.status_label.config(text=f"Output folder set: {folder}")

    def _refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        for path in self.selected_paths:
            self.file_listbox.insert(tk.END, Path(path).name)
        self.status_label.config(text=f"{len(self.selected_paths)} file(s) selected.")

    # ------------------------------------------------------------------
    # OCR execution
    # ------------------------------------------------------------------
    def run_ocr(self):
        if not self.selected_paths:
            messagebox.showwarning("No files", "Select PDF file(s) or a folder first.")
            return

        self.progress["maximum"] = len(self.selected_paths)
        self.progress["value"] = 0
        self.status_label.config(text="Running OCR... (first run downloads models, can take a while)")

        thread = threading.Thread(target=self._run_ocr_worker, daemon=True)
        thread.start()

    def _run_ocr_worker(self):
        for i, path in enumerate(self.selected_paths, start=1):
            filename = Path(path).name
            try:
                result = process_pdf(path)
                self.results[filename] = result
            except Exception as e:
                # Don't let one bad PDF kill the whole batch.
                error_trace = traceback.format_exc()
                self.results[filename] = {"file": filename, "error": str(e), "trace": error_trace}

            self.root.after(0, self._update_progress, i, filename)

        self.root.after(0, self._on_ocr_complete)

    def _update_progress(self, count, filename):
        self.progress["value"] = count
        self.status_label.config(text=f"Processed {count}/{len(self.selected_paths)}: {filename}")

    def _on_ocr_complete(self):
        self.status_label.config(text=f"Done. {len(self.results)} file(s) processed.")
        # Auto-select the first file so the user sees output immediately
        if self.file_listbox.size() > 0:
            self.file_listbox.selection_set(0)
            self.on_file_select(None)

    # ------------------------------------------------------------------
    # Field extraction (qwen33b via Ollama) + Excel export
    # ------------------------------------------------------------------
    def run_extraction(self):
        # Only consider files that actually OCR'd successfully -- an
        # "error" result has no markdown to extract from.
        ocr_ok = {fn: r for fn, r in self.results.items() if "error" not in r}

        if not ocr_ok:
            messagebox.showwarning("Nothing to extract", "Run OCR successfully on at least one file first.")
            return

        out_dir = self._get_output_dir()
        if out_dir is None:
            return

        self.progress["maximum"] = len(ocr_ok)
        self.progress["value"] = 0
        self.status_label.config(text="Extracting fields via qwen33b... (this calls Ollama per file)")

        thread = threading.Thread(target=self._run_extraction_worker, args=(ocr_ok, out_dir), daemon=True)
        thread.start()

    def _run_extraction_worker(self, ocr_ok, out_dir):
        for i, (filename, ocr_result) in enumerate(ocr_ok.items(), start=1):
            try:
                doc_markdown = combined_markdown(ocr_result)
                extraction = extract(doc_markdown, source_file=filename)
            except Exception as e:
                # Same principle as OCR: one bad file (e.g. Ollama down,
                # bad JSON from the model) shouldn't kill the whole batch.
                error_trace = traceback.format_exc()
                extraction = {
                    "source_file": filename,
                    "is_invoice": None,
                    "document_type": "EXTRACTION_ERROR",
                    "fields": {},
                    "needs_review": True,
                    "review_reasons": [f"Extraction failed: {e}"],
                    "raw_model_response": error_trace,
                }
            self.extraction_results[filename] = extraction
            self.root.after(0, self._update_progress, i, filename)

        try:
            xlsx_path = out_dir / "invoices_output.xlsx"
            build_workbook(list(self.extraction_results.values()), str(xlsx_path))
            self.root.after(0, self._on_extraction_complete, str(xlsx_path), None)
        except Exception as e:
            self.root.after(0, self._on_extraction_complete, None, str(e))

    def _on_extraction_complete(self, xlsx_path, error):
        review_count = sum(1 for r in self.extraction_results.values() if r.get("needs_review"))
        total = len(self.extraction_results)

        if error:
            self.status_label.config(text=f"Extraction done, but Excel export failed: {error}")
            messagebox.showerror("Excel export failed", error)
            return

        self.status_label.config(text=f"Extracted {total} file(s), {review_count} need review. Saved to {xlsx_path}")
        messagebox.showinfo(
            "Extraction complete",
            f"Processed {total} file(s).\n{review_count} row(s) flagged for review.\n\nSaved: {xlsx_path}",
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        filename = self.file_listbox.get(selection[0])
        result = self.results.get(filename)

        self.text_output.delete("1.0", tk.END)

        if result is None:
            self.text_output.insert(tk.END, "(Not yet processed - click Run OCR)")
            return

        if "error" in result:
            self.text_output.insert(tk.END, f"ERROR processing {filename}:\n{result['error']}\n\n{result.get('trace', '')}")
            return

        self.text_output.insert(tk.END, combined_markdown(result))

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _get_output_dir(self):
        if self.output_dir is None:
            folder = filedialog.askdirectory(title="Select output folder")
            if not folder:
                return None
            self.output_dir = Path(folder)
        return self.output_dir

    def save_selected(self, fmt):
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("No selection", "Select a file from the list first.")
            return
        filename = self.file_listbox.get(selection[0])
        result = self.results.get(filename)
        if result is None:
            messagebox.showwarning("Not processed", "Run OCR before saving.")
            return
        self._save_one(filename, result, fmt)

    def save_all(self, fmt):
        if not self.results:
            messagebox.showwarning("Nothing to save", "Run OCR before saving.")
            return
        out_dir = self._get_output_dir()
        if out_dir is None:
            return
        for filename, result in self.results.items():
            self._save_one(filename, result, fmt, out_dir=out_dir)
        messagebox.showinfo("Saved", f"Saved {len(self.results)} file(s) to {out_dir}")

    def _save_one(self, filename, result, fmt, out_dir=None):
        if out_dir is None:
            out_dir = self._get_output_dir()
            if out_dir is None:
                return

        stem = Path(filename).stem

        if fmt == "json":
            out_path = out_dir / f"{stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                # result["pages"][i]["json"] contains numpy arrays (coordinates,
                # scores) from the pipeline's own result object, which
                # json.dump() can't serialize directly. default=str keeps
                # this simple slice from crashing -- if you need those arrays
                # as real numbers later, that's worth a proper serializer,
                # not a quick str() fallback.
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        else:  # md
            out_path = out_dir / f"{stem}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                if "error" in result:
                    f.write(f"ERROR: {result['error']}\n")
                else:
                    f.write(combined_markdown(result))

        self.status_label.config(text=f"Saved {out_path.name}")


def main():
    root = tk.Tk()
    app = InvoiceraOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
