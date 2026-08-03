"""
extract_fields.py

Stage 2 of the Invoicera pipeline: takes the markdown produced by
ocr_pipeline.py's process_page()/process_pdf() and extracts structured
invoice fields using a local Ollama model (default: "qwen33b").

Design goals (per discussion):
- Minimize what the LLM has to infer. Feed it clean, structured markdown
  (not raw OCR JSON), a fixed schema, and few-shot examples built from
  real invoices -- not a generic "extract the invoice" prompt.
- Never let the model guess. Every field must be null if not confidently
  present, and every non-null field must carry a verbatim `source_text`
  snippet so we can check it's grounded in the actual document instead
  of invented.
- Detect non-invoice documents (e.g. "Note d'Honoraires") explicitly as
  part of the same call, using a document_type field -- not as an
  afterthought. Non-invoice docs get routed to a separate schema/sheet.
- Validate everything a second time in Python before trusting it:
  regex sanity checks, source_text grounding, arithmetic reconciliation.
  A field that fails validation is flagged, never silently corrected or
  silently trusted.

This module does NOT talk to PP-StructureV3 directly -- it consumes the
`markdown` string that ocr_pipeline.process_page()/process_pdf() already
produces. Wire it in downstream of that.

ASSUMPTIONS (flagging per coding-principles -- confirm these against your
real invoices):
- FOURNISSEUR = the issuing/letterhead company, never "ROCHE ALGERIE SPA"
  (confirmed: Roche is always the client in your samples).
- The "PO INVOICE" number (e.g. 1640117789) is captured separately from
  the invoice's own N degrees, for cross-checking against the source filename.
- Ollama is reachable at the default http://localhost:11434. Change
  OLLAMA_URL if yours differs.
- Model name is "qwen33b" as you named it locally -- change MODEL_NAME
  if you rename it.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Optional


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen33b"

# Ollama call is deterministic-leaning on purpose: this is extraction,
# not creative writing. Lower temperature = less room to hallucinate
# plausible-looking but invented values.
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": 8192,
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# Fields that must appear in the returned JSON for an invoice ("facture").
# Every value is either a string/number or null -- never omitted, so
# downstream code can rely on the key always being present.
INVOICE_FIELDS = [
    "RC",
    "NIF",
    "DATE",
    "FOURNISSEUR",
    "N_FACTURE",
    "PO_INVOICE",
    "MONTANT_HT",
    "TVA",
    "MONTANT_TTC",
]

# Separate schema for non-invoice documents (e.g. Note d'Honoraires).
# Deliberately different shape -- these go to a separate sheet, not
# force-fit into the invoice columns.
OTHER_DOC_FIELDS = [
    "DATE",
    "EMETTEUR",          # who issued the document (e.g. "Pr. BENLAHRACH Zakia")
    "DESTINATAIRE",       # who it's addressed to
    "REFERENCE",
    "MONTANT_BRUT",
    "RETENUE",            # e.g. IRG deduction, if present
    "NET_A_PAYER",
    "PO_INVOICE",
]


SYSTEM_PROMPT = f"""You are a strict, literal document field extractor for Algerian business documents (invoices, honorarium notes, etc). You NEVER guess or infer a value that is not clearly present in the text. If a field is not clearly present, its value MUST be null.

You will receive the OCR'd markdown text of ONE document. The OCR may have minor errors: a digit or letter misread (0/O, 1/l), or missing spaces between words (e.g. "Designationduredevable" instead of "Designation du redevable"). Do your best to read through these artifacts, but do not invent data that isn't there to compensate for them.

STEP 1 -- Classify the document:
- "is_invoice": true if this is a standard commercial invoice/facture with an itemized table (quantities, unit prices) and a HT/TVA/TTC-style total breakdown.
- "is_invoice": false for anything else (honorarium notes, notes d'honoraires, letters, receipts without itemization, anything that doesn't have that structure). Do NOT force such documents into the invoice schema.
- "document_type": a short label, e.g. "facture", "note_honoraires", "autre".

STEP 2 -- Extract fields based on classification:
- If is_invoice is true, fill this exact schema: {json.dumps(INVOICE_FIELDS)}
- If is_invoice is false, fill this exact schema instead: {json.dumps(OTHER_DOC_FIELDS)}
- Every key in the relevant schema MUST appear in your output. Use null for anything not clearly present. Never omit a key.
- FOURNISSEUR (invoices only) is the ISSUING company shown in the letterhead/header -- the one whose RC/NIF and bank details appear at the bottom or top of the page. It is NEVER "Roche Algerie SPA" -- that is always the client, not the supplier.
- PO_INVOICE is the number printed near/under the "PO INVOICE" label (a plain digit string), distinct from the invoice's own N degrees / FACTURE N degrees.
- For every non-null field, also add a matching "<field>_source" key containing the EXACT verbatim text snippet (a short substring, not reworded) from the input that you took the value from. This must be copy-pasted, not paraphrased -- it's used to verify you didn't invent the value.
- Dates: normalize to DD/MM/YYYY if the source format is unambiguous, otherwise return exactly as printed.
- Amounts: return as plain numbers (no thousand separators, no currency symbol, use "." for decimals). If a value is genuinely absent (e.g. TVA is "NON ASSUJETTI A LA TVA" / not subject to VAT), use null and note this is a valid null, not a miss.

Respond with ONLY a single JSON object, no markdown fences, no commentary, no preamble. Shape:
{{
  "is_invoice": true|false,
  "document_type": "...",
  "fields": {{ ... }}
}}
"""


# Few-shot examples embedded in the prompt. Built from the real sample
# invoices you sent. Keep these short -- they cost context on every call.
# NOTE: these are illustrative excerpts, not full documents, to keep the
# prompt lean. Expand with more real examples (especially edge cases) as
# you find failure modes.
FEW_SHOT_EXAMPLES = [
    {
        "input": """Facture N°: FC260118
Date Facture : 25/05/26
PO INVOICE
1640117789
Nom Client: SPA ROCHE ALGERIE
RC N° :
MF :
Total HT : 19 500,00
TVA 19 % : -
Total TTC : 19 500,00
Raison Sociale : MAMME ABDELFATAH   Adresse Siege : COOP J8 N° 162 KOUBA ALGER
N° RC : 21 A 5910273-16/01   NIF 19339110124317311601""",
        "output": {
            "is_invoice": True,
            "document_type": "facture",
            "fields": {
                "RC": "21 A 5910273-16/01",
                "RC_source": "N° RC : 21 A 5910273-16/01",
                "NIF": "19339110124317311601",
                "NIF_source": "NIF 19339110124317311601",
                "DATE": "25/05/2026",
                "DATE_source": "Date Facture : 25/05/26",
                "FOURNISSEUR": "MAMME ABDELFATAH",
                "FOURNISSEUR_source": "Raison Sociale : MAMME ABDELFATAH",
                "N_FACTURE": "FC260118",
                "N_FACTURE_source": "Facture N°: FC260118",
                "PO_INVOICE": "1640117789",
                "PO_INVOICE_source": "PO INVOICE\n1640117789",
                "MONTANT_HT": 19500.00,
                "MONTANT_HT_source": "Total HT : 19 500,00",
                "TVA": None,
                "TVA_source": None,
                "MONTANT_TTC": 19500.00,
                "MONTANT_TTC_source": "Total TTC : 19 500,00",
            },
        },
        "note": "Note the RC field on the client side (blank, unfilled 'RC N° :') is correctly ignored -- FOURNISSEUR's own RC at the bottom is used instead. TVA is null because the stamp says NON ASSUJETTI A LA TVA (not subject to VAT) -- a real, valid null, not a missed field.",
    },
    {
        "input": """22/05/2026 09:52  Note d'honoraires - " Parcours et Defis..."
Alger, le 23/05/2026
A l'attention de
Roche Algerie SPA
PO INVOICE
1640117822
Reference : CT2605AA6566
NOTE D'HONORAIRES
Honoraires (Montant brut)   164705.88
IRG (15%) retenus a la source   24705.88
Total Net a payer   140000.00
Pr. BENLAHRACH Zakia
Etablissement Public Hospitalier de Laghouat""",
        "output": {
            "is_invoice": False,
            "document_type": "note_honoraires",
            "fields": {
                "DATE": "23/05/2026",
                "DATE_source": "Alger, le 23/05/2026",
                "EMETTEUR": "Pr. BENLAHRACH Zakia",
                "EMETTEUR_source": "Pr. BENLAHRACH Zakia",
                "DESTINATAIRE": "Roche Algerie SPA",
                "DESTINATAIRE_source": "A l'attention de\nRoche Algerie SPA",
                "REFERENCE": "CT2605AA6566",
                "REFERENCE_source": "Reference : CT2605AA6566",
                "MONTANT_BRUT": 164705.88,
                "MONTANT_BRUT_source": "Honoraires (Montant brut)   164705.88",
                "RETENUE": 24705.88,
                "RETENUE_source": "IRG (15%) retenus a la source   24705.88",
                "NET_A_PAYER": 140000.00,
                "NET_A_PAYER_source": "Total Net a payer   140000.00",
                "PO_INVOICE": "1640117822",
                "PO_INVOICE_source": "PO INVOICE\n1640117822",
            },
        },
        "note": "This document has no itemized table and no HT/TVA/TTC structure -- correctly classified is_invoice=false and routed to the OTHER_DOC_FIELDS schema instead of being force-fit into invoice fields.",
    },
]


def _build_prompt(document_markdown: str) -> str:
    examples_text = "\n\n".join(
        f"EXAMPLE INPUT:\n{ex['input']}\n\nEXAMPLE OUTPUT:\n{json.dumps(ex['output'], ensure_ascii=False, indent=2)}\n\n({ex['note']})"
        for ex in FEW_SHOT_EXAMPLES
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Here are worked examples:\n\n{examples_text}\n\n"
        f"---\n\nNow extract from this document:\n\n{document_markdown}\n\n"
        f"Respond with ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str = MODEL_NAME, url: str = OLLAMA_URL, timeout: int = 300) -> str:
    """
    Calls a local Ollama instance's /api/generate endpoint (non-streaming).
    Returns the raw text response. Raises RuntimeError on connection or
    HTTP failure with a message pointing at what to check.

    NOT VERIFIED against a live Ollama instance in this environment --
    this sandbox has no network path to your machine's localhost. Test
    this function directly first (see __main__ block below) before
    trusting the rest of the pipeline.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": OLLAMA_OPTIONS,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    # Diagnostic: prompt size going out. If this number is high relative
    # to OLLAMA_OPTIONS["num_ctx"] (currently 8192), that's the first
    # thing to suspect on a timeout/hang -- an over-context prompt doesn't
    # always fail cleanly, it can just grind.
    approx_tokens = len(prompt) // 4
    print(f"[extract_fields] Calling Ollama model='{model}', prompt chars={len(prompt)}, approx tokens={approx_tokens}, timeout={timeout}s")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except TimeoutError as e:
        raise RuntimeError(
            f"Ollama call timed out after {timeout}s (prompt was ~{approx_tokens} tokens). "
            f"If this keeps happening, check the Ollama server's own console/logs for what "
            f"it was doing when this hung -- a silent generation loop or a context-window "
            f"issue won't necessarily show up as an HTTP-level error."
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {url}. Is `ollama serve` running, "
            f"and is the model name '{model}' correct (check `ollama list`)? "
            f"Underlying error: {e}"
        ) from e

    if "response" not in body:
        raise RuntimeError(f"Unexpected Ollama response shape (no 'response' key): {body}")

    return body["response"]


def _strip_json_fences(text: str) -> str:
    """Ollama models sometimes wrap JSON in ```json fences despite instructions not to. Strip if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Loose sanity patterns -- NOT used to reject/invent values, only to flag
# suspicious ones for review. Your real RC formats vary too much (see
# samples: "21 A 5910273-16/01", "16/005052480 A21", "31/01-0813013A98",
# "0342127-A05") for a strict single regex, so this only checks for the
# presence of both digits and enough length to look plausible.
_RC_PLAUSIBLE = re.compile(r"\d{4,}")
_NIF_PLAUSIBLE = re.compile(r"^\d{10,20}$")


@dataclass
class ValidationResult:
    needs_review: bool = False
    reasons: list[str] = field(default_factory=list)

    def flag(self, reason: str) -> None:
        self.needs_review = True
        self.reasons.append(reason)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


def validate_invoice_fields(fields: dict, document_markdown: str) -> ValidationResult:
    """
    Second-pass, non-LLM validation. Never silently corrects a value --
    only flags it. This is what actually catches OCR-induced or model
    hallucination errors before they reach the Excel sheet.
    """
    result = ValidationResult()

    # 1. Grounding check: does every claimed source_text actually appear
    #    (allowing for whitespace differences) in the source document?
    #    This is the single strongest hallucination catch -- a value the
    #    model invented will not have a real source_text to point to.
    normalized_doc = re.sub(r"\s+", " ", document_markdown).lower()
    for f in INVOICE_FIELDS:
        value = fields.get(f)
        source = fields.get(f"{f}_source")
        if value is None:
            continue
        if not source:
            result.flag(f"{f} has a value but no source_text -- ungrounded, treat as suspect")
            continue
        normalized_source = re.sub(r"\s+", " ", str(source)).lower()
        if normalized_source not in normalized_doc:
            result.flag(f"{f}_source ('{source}') not found verbatim in document -- possible hallucination")

    # 2. RC plausibility (loose -- see _RC_PLAUSIBLE note above)
    rc = fields.get("RC")
    if rc and not _RC_PLAUSIBLE.search(str(rc)):
        result.flag(f"RC value '{rc}' doesn't look like a plausible RC (no digit run found)")

    # 3. NIF plausibility
    nif = fields.get("NIF")
    if nif and not _NIF_PLAUSIBLE.match(str(nif).replace(" ", "")):
        result.flag(f"NIF value '{nif}' doesn't match expected all-digit, 10-20 char pattern")

    # 4. Arithmetic reconciliation: HT + TVA ~= TTC (only when both HT and
    #    TTC are present; TVA can legitimately be null for NON ASSUJETTI
    #    cases, so only check when TVA also has a value).
    ht = _to_float(fields.get("MONTANT_HT"))
    tva = _to_float(fields.get("TVA"))
    ttc = _to_float(fields.get("MONTANT_TTC"))
    if ht is not None and ttc is not None:
        if tva is not None:
            expected_ttc = ht + tva
            if abs(expected_ttc - ttc) > 1.0:  # 1 DA tolerance for rounding
                result.flag(f"HT ({ht}) + TVA ({tva}) = {expected_ttc}, does not match TTC ({ttc})")
        elif abs(ht - ttc) > 1.0:
            # TVA null (e.g. NON ASSUJETTI) -- HT should equal TTC in that case
            result.flag(f"TVA is null but HT ({ht}) != TTC ({ttc}) -- expected them equal when not subject to VAT")

    return result


def validate_other_doc_fields(fields: dict, document_markdown: str) -> ValidationResult:
    """Same grounding check, applied to the non-invoice schema."""
    result = ValidationResult()
    normalized_doc = re.sub(r"\s+", " ", document_markdown).lower()
    for f in OTHER_DOC_FIELDS:
        value = fields.get(f)
        source = fields.get(f"{f}_source")
        if value is None:
            continue
        if not source:
            result.flag(f"{f} has a value but no source_text -- ungrounded, treat as suspect")
            continue
        normalized_source = re.sub(r"\s+", " ", str(source)).lower()
        if normalized_source not in normalized_doc:
            result.flag(f"{f}_source ('{source}') not found verbatim in document -- possible hallucination")

    brut = _to_float(fields.get("MONTANT_BRUT"))
    retenue = _to_float(fields.get("RETENUE"))
    net = _to_float(fields.get("NET_A_PAYER"))
    if brut is not None and retenue is not None and net is not None:
        if abs((brut - retenue) - net) > 1.0:
            result.flag(f"MONTANT_BRUT ({brut}) - RETENUE ({retenue}) != NET_A_PAYER ({net})")

    return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def extract(document_markdown: str, source_file: str = "", model: str = MODEL_NAME) -> dict:
    """
    Full stage-2 pipeline for one document's markdown text (one invoice,
    already OCR'd by process_page/process_pdf).

    Returns a dict:
        {
            "source_file": str,
            "is_invoice": bool,
            "document_type": str,
            "fields": {...},                # raw fields from the model
            "needs_review": bool,
            "review_reasons": [...],
            "raw_model_response": str,       # kept for debugging bad parses
        }
    """
    prompt = _build_prompt(document_markdown)
    raw_response = call_ollama(prompt, model=model)
    cleaned = _strip_json_fences(raw_response)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Model didn't return valid JSON -- don't guess-fix it, surface
        # the failure so it's visibly flagged rather than silently empty.
        return {
            "source_file": source_file,
            "is_invoice": None,
            "document_type": "PARSE_ERROR",
            "fields": {},
            "needs_review": True,
            "review_reasons": [f"Model response was not valid JSON: {e}"],
            "raw_model_response": raw_response,
        }

    is_invoice = parsed.get("is_invoice")
    document_type = parsed.get("document_type", "unknown")
    fields = parsed.get("fields", {})

    if is_invoice:
        validation = validate_invoice_fields(fields, document_markdown)
    else:
        validation = validate_other_doc_fields(fields, document_markdown)

    return {
        "source_file": source_file,
        "is_invoice": is_invoice,
        "document_type": document_type,
        "fields": fields,
        "needs_review": validation.needs_review,
        "review_reasons": validation.reasons,
        "raw_model_response": raw_response,
    }


if __name__ == "__main__":
    # Quick manual test against a live Ollama instance. Run this directly
    # on your machine (not in this sandbox) to verify the model call
    # works before wiring it into gui.py:
    #
    #   python extract_fields.py
    #
    # Paste real markdown output from process_page() below, or import
    # this module and call extract() with real pipeline output.
    sample_markdown = """
Facture N°: FC260118
Date Facture : 25/05/26
Nom Client: SPA ROCHE ALGERIE
Total HT : 19 500,00
Total TTC : 19 500,00
N° RC : 21 A 5910273-16/01   NIF 19339110124317311601
Raison Sociale : MAMME ABDELFATAH
"""
    result = extract(sample_markdown, source_file="test.pdf")
    print(json.dumps(result, indent=2, ensure_ascii=False))
