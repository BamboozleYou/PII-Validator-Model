import re
import json
import logging
import requests
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# ---------- Logging ----------
logger = logging.getLogger("pii_validator")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)  # bumped to DEBUG when cfg.debug=True

# --- Config ---------------------------------------------------

@dataclass
class ModelConfig:
    provider: str = "none"         # "ollama", "openai", or "none"
    model: str = ""
    base_url: str = "http://localhost:11434"
    label_space: Optional[List[str]] = None
    request_timeout: int = 45
    debug: bool = False            # enable verbose logs

# --- Fixed label set (use these EXACT human labels) ----------

PII_LABELS: List[str] = [
    "Address",
    "Address 2",
    "Age",
    "Bank account",
    "City",
    "Country",
    "Date of birth",
    "Drivers license number",
    "Email addresses",
    "First name",
    "Full name",
    "Insurance number",
    "Last name",
    "Medical records",
    "Medicare ID",
    "National Identifier",
    "Phone",
    "Postal Code",
    "Date of Death",
    "Insurance Number",
]

def _column_tokens(name: str) -> List[str]:
    """Canonical token list, e.g., 'customer_phone' -> ['customer','phone']"""
    norm = _normalize_colname(name)
    return [t for t in norm.split() if t]

# --- Column name canonicalization (normalize + tokenize) -----
_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])')

# at top: _CAMEL_RE already defined
def _normalize_colname(name: str) -> str:
    """Normalize a column name for analysis (lowercase, spaces between parts)."""
    if not name:
        return ""
    s = str(name)
    s = re.sub(r'[\[\]]', '', s)                       # [inp_PSSN] -> inp_PSSN
    s = re.sub(r'^(inp_|pd_|tbl_|usr_|cust_)', '', s)  # drop common tech prefixes
    s = _CAMEL_RE.sub(" ", s)                          # userEmail2FA -> user Email 2 FA
    # split glued ALL-CAPS tokens on common suffixes/abbrs (FIX: pass 's' as 3rd arg)
    s = re.sub(
        r'(?i)(?<=\w)(name|code|date|time|status|flag|count|number|complaint|note|notes|'
        r'dob|ssn|fname|lname|mname|dfname|dlname|pfn|pln|sfn|sln)\b',
        r' \1',
        s
    )
    s = re.sub(r'[_\W]+', ' ', s)                      # underscores/punct -> spaces
    return " ".join(s.split()).lower()

NOISE_TOKENS = {
    "etl", "anc", "datasource", "source", "onesource", "data", "dataset",
    "table", "tbl", "column", "col", "field"
}

# add this near the top with your other helpers
TOKEN_HINTS = {
    # generic identifiers
    "dob": "Date of birth",
    "ssn": "National identifier",
    "name": "Full name or a name component (person, not address)",
    "fname": "First name",
    "lname": "Last name",
    "mname": "Middle name (name component, not a final label here)",
    # your sheet’s common abbreviations
    "pdob": "Date of birth",
    "sdob": "Date of birth",
    "dfname": "First name",
    "dlname": "Last name",
    "pfn": "First name",
    "pln": "Last name",
    "sfn": "First name",
    "sln": "Last name",
    # entities (disambiguate away from Address)
    "patient": "Person/patient indicator (medical records, not address)",
    "customer": "Person/customer indicator (person, not address)",
    # insurance/policy
    "policy": "Insurance/policy identifier",
    "ins": "Insurance",
}
def _token_glossary(tokens: List[str]) -> str:
    seen = []
    for t in tokens:
        hint = TOKEN_HINTS.get(t)
        if hint:
            seen.append(f"- {t} → {hint}")
    return "\n".join(seen)


def _token_glossary(tokens: List[str]) -> str:
    """Make a small, column-specific glossary to help the SLM."""
    seen = []
    for t in tokens:
        hint = TOKEN_HINTS.get(t)
        if hint:
            seen.append(f"- {t} → {hint}")
    return "\n".join(seen)

# -- Guide to use in the prompt (for the SLM to learn from) -- #

PII_GUIDE = {
    "Address": {
        "desc": "Street/mailing address line 1.",
        "patterns": ["street", "road", "lane", "avenue", "addr", "address", "saddress", "daddress", "home_address"],
        "types": ["string", "varchar", "text"],
        "length": "5–200 chars; free text (may include numbers and punctuation)"
    },
    "Address 2": {
        "desc": "Supplemental address line (apartment/suite/flat).",
        "patterns": ["address 2", "addr2", "line 2", "suite", "apt", "apartment", "flat"],
        "types": ["string", "varchar", "text"],
        "length": "1–100 chars"
    },
    "Age": {
        "desc": "Person's age in years.",
        "patterns": ["age"],
        "types": ["int", "integer", "smallint", "tinyint", "number"],
        "length": "1–3 digits (0–120 typical)"
    },
    "Bank account": {
        "desc": "Bank/financial account number (digits or alphanumeric; may include spaces).",
        "patterns": ["account", "acct", "iban", "accno"],
        "types": ["string", "varchar", "text", "bigint"],
        "length": "8–34 chars typical (IBAN up to 34)"
    },
    "City": {
        "desc": "City or town name.",
        "patterns": ["city", "town", "municipality"],
        "types": ["string", "varchar", "text"],
        "length": "2–85 chars"
    },
    "Country": {
        "desc": "Country name or ISO code.",
        "patterns": ["country"],
        "types": ["string", "varchar", "text", "char"],
        "length": "2–3 chars for code (e.g., IN/USA) or full country name"
    },
    "Date of birth": {
        "desc": "Birth date.",
        "patterns": ["dob", "birth", "dateofbirth", "ddob", "birth_date", "birthdate"],
        "types": ["date", "datetime", "timestamp", "string"],
        "length": "8–10+ depending on format (e.g., YYYY-MM-DD, DD/MM/YYYY)"
    },
    "Drivers license number": {
        "desc": "Government driver licence/ID number.",
        "patterns": ["dl", "licence", "license", "driving"],
        "types": ["string", "varchar", "text"],
        "length": "6–18 alphanumeric typical"
    },
    "Email addresses": {
        "desc": "Email address.",
        "patterns": ["email", "e-mail", "mail", "user_email", "customer_email", "email_addr"],
        "types": ["string", "varchar", "text"],
        "length": "6–254 chars; contains '@' and domain"
    },
    "First name": {
        "desc": "Given/first name.",
        "patterns": ["first", "given", "fname", "sfname", "sfn", "firstname", "dfname"],
        "types": ["string", "varchar", "text"],
        "length": "1–50 chars"
    },
    "Full name": {
        "desc": "Full personal name (may include spaces).",
        "patterns": ["name", "full name", "employee_name", "customer_name", "fullname"],
        "types": ["string", "varchar", "text"],
        "length": "5–100 chars typical"
    },
    "Insurance number": {
        "desc": "Insurance/policy/member/subscriber ID.",
        "patterns": ["insurance", "policy", "member", "subscriber"],
        "types": ["string", "varchar", "text"],
        "length": "8–20 alphanumeric typical"
    },
    "Last name": {
        "desc": "Family/surname/last name.",
        "patterns": ["last", "surname", "lname", "family", "slname", "sln", "lastname", "dlname", "dmname"],
        "types": ["string", "varchar", "text"],
        "length": "1–50 chars"
    },
    "Medical records": {
        "desc": "Medical record identifiers (e.g., MRN) or fields containing clinical info.",
        "patterns": ["medical", "mrn", "ehr", "chart", "clinical", "health"],
        "types": ["string", "varchar", "text"],
        "length": "varies; MRNs often 6–12"
    },
    "Medicare ID": {
        "desc": "Medicare programme identifier.",
        "patterns": ["medicare", "hic", "hic_id", "mco_name", "mco_plan", "medicare_id", "medicare_num", "health_insurance_claim"],
        "types": ["string", "varchar", "text"],
        "length": "region-specific; often 10–11 alphanumeric, but can vary"
    },
    "National Identifier": {
        "desc": "Government-issued citizen ID (e.g., SSN, Aadhaar, PAN, NRIC).",
        "patterns": ["ssn", "aadhaar", "aadhar", "pan", "nric", "nin", "nid", "national id", "identifier", "pssn", "sssn", "dssn", "social_security"],
        "types": ["string", "varchar", "text"],
        "length": "country-specific (e.g., SSN 9 digits; Aadhaar 12; PAN 10 alphanum)"
    },
    "Phone": {
        "desc": "Telephone or mobile number (may include + country code).",
        "patterns": ["phone", "mobile", "telephone", "tel", "cell", "msisdn", "customer_phone", "mobile_number", "phone_number"],
        "types": ["string", "varchar", "text", "bigint"],
        "length": "7–15 digits typical (may include + and separators)"
    },
    "Postal Code": {
        "desc": "Postal/ZIP/PIN code.",
        "patterns": ["postal", "zip", "postcode", "pincode", "zip_code", "postal_code"],
        "types": ["string", "varchar", "text", "int"],
        "length": "4–10 alphanumeric depending on country (IN 6 digits, US 5–10)"
    },
    "Date of death": {
    "desc": "Death date.",
    "patterns": ["dod", "death", "date_of_death", "deathdate", "death_date"],
    "types": ["date", "datetime", "timestamp", "string"],
    "length": "8–10+ depending on format (e.g., YYYY-MM-DD, DD/MM/YYYY)"
},

}

def _build_guide_text(label_space: List[str]) -> str:
    lines = []
    for lbl in label_space:
        g = PII_GUIDE.get(lbl, {})
        if not g:
            continue
        desc = g.get("desc", "")
        pats = ", ".join(g.get("patterns", []))
        typs = ", ".join(g.get("types", []))
        lhint = g.get("length", "")
        lines.append(f"- {lbl}: {desc} | Patterns: {pats} | Types: {typs} | Length: {lhint}")
    return "\n".join(lines)

# --- Canonicalization of SLM output (NOT pre-classification) -

def _canonicalize(label: str, label_space: List[str]) -> Optional[str]:
    """
    Map the SLM's classification string to a canonical label in label_space.
    No heuristics over column names here — only normalize the returned label.
    """
    if not label:
        return None
    txt = label.strip()
    if txt.lower() == "unsure":
        return None
    norm = txt.lower()

    synonyms = {
        # Medicare
        "medicare id": "Medicare ID",
        "medicare": "Medicare ID",
        "hic": "Medicare ID",
        "hic id": "Medicare ID",
        "hic_id": "Medicare ID",
        "mco name": "Medicare ID",
        "mco plan": "Medicare ID",
        "mco_name": "Medicare ID",
        "mco_plan": "Medicare ID",
        "health insurance claim": "Medicare ID",

        # Email
        "email": "Email addresses",
        "email address": "Email addresses",
        "emails": "Email addresses",
        "e-mail": "Email addresses",
        "mail": "Email addresses",
        "email addr": "Email addresses",

        # Phone
        "phone number": "Phone",
        "telephone": "Phone",
        "mobile": "Phone",
        "cell": "Phone",
        "phone no": "Phone",
        "tel": "Phone",
        "mobile number": "Phone",
        "cell phone": "Phone",
        "telephone number": "Phone",

        # DOB
        "dob": "Date of birth",
        "date of birth": "Date of birth",
        "birthdate": "Date of birth",
        "birth date": "Date of birth",
        "birth_date": "Date of birth",
        "dateofbirth": "Date of birth",

        # Postal
        "postal code": "Postal Code",
        "postcode": "Postal Code",
        "zip": "Postal Code",
        "zip code": "Postal Code",
        "zipcode": "Postal Code",
        "zip_code": "Postal Code",
        "pin code": "Postal Code",
        "pincode": "Postal Code",

        # License
        "drivers license": "Drivers license number",
        "driver license": "Drivers license number",
        "driving license": "Drivers license number",
        "license": "Drivers license number",
        "licence": "Drivers license number",

        # Address
        "address": "Address",
        "addr": "Address",

        "address line 2": "Address 2",
        "addr2": "Address 2",
        "address2": "Address 2",

        # Names
        "first name": "First name",
        "fname": "First name",
        "firstname": "First name",
        "given name": "First name",
        "forename": "First name",

        "last name": "Last name",
        "lname": "Last name",
        "lastname": "Last name",
        "surname": "Last name",
        "family name": "Last name",

        "full name": "Full name",
        "complete name": "Full name",
        "name": "Full name",

        # National ID
        "ssn": "National Identifier",
        "social security number": "National Identifier",
        "social security": "National Identifier",
        "aadhaar": "National Identifier",
        "aadhar": "National Identifier",
        "pan": "National Identifier",
        "national id": "National Identifier",
        "citizen id": "National Identifier",

        # Other direct
        "age": "Age",
        "city": "City",
        "town": "City",
        "country": "Country",

       # Date of death variations
        "dod": "Date of death",
        "date of death": "Date of death",
        "death date": "Date of death",
        "deathdate": "Date of death",
        "death_date": "Date of death",

        # Bank
        "account": "Bank account",
        "bank account": "Bank account",
        "account number": "Bank account",
        # Insurance number variations
        "insurance no": "Insurance number",
        "insurance number": "Insurance number",
        "ins number": "Insurance number",
        "policy id": "Insurance number",
        "policy number": "Insurance number",
        "member id": "Insurance number",
        "subscriber id": "Insurance number",

    }

    mapped = synonyms.get(norm)
    if mapped and mapped in label_space:
        return mapped

    # Case-insensitive exact match
    for l in label_space:
        if l.lower() == norm:
            return l

    # Light plural cleanup
    s2 = norm[:-1] if norm.endswith("s") and not norm.endswith("ss") else norm
    for l in label_space:
        l2 = l.lower()[:-1] if l.lower().endswith("s") and not l.lower().endswith("ss") else l.lower()
        if l2 == s2:
            return l

    return None

# --- Model Client (SLM does all the classification) ----------

class ModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        if getattr(self.cfg, "debug", False):
            logger.setLevel(logging.DEBUG)

    def _log(self, msg: str):
        if self.cfg.debug:
            logger.debug(msg)

    def query(
        self,
        column_name: str,
        datatype: str,
        col_length: int,
        label_space: List[str]
    ) -> Tuple[Optional[str], str]:
        """
        Returns (classification, reasoning), where classification is a canonical
        label from label_space (or None for Unsure). 
        """
        if self.cfg.provider == "none":
            logger.warning("Model provider is 'none' — set --provider ollama/openai and --model <n> to enable SLM.")
            return None, ""

        labels_csv = ", ".join(label_space)
        normalized = _normalize_colname(column_name)
        tokens = _column_tokens(column_name)
        guide_text = _build_guide_text(label_space)
        glossary = _token_glossary(tokens) or "(no special abbreviations detected)"

        dtype_text = str(datatype) if datatype is not None else "unknown"
        length_text = str(col_length) if col_length is not None else "unknown"

        prompt = (
            "You are a STRICT PII column classifier.\n"
            "TASK: Given a database column, choose EXACTLY ONE label from Allowed labels.\n"
            "If the field is metadata ABOUT PII (e.g., status/flag/code/date/verified/updated), or it clearly does not fit a label, return exactly 'Unsure'.\n\n"

            "## How to read names\n"
            "- Use the normalized name and the token list.\n"
            "- Tokens are produced by canonicalizing the raw column name (camelCase/underscores/ALL-CAPS splits).\n\n"
            """## Decision Rules (follow in order)
            1) Decide from column-name meaning using the tokens.
            2) If name indicates metadata (suffixes like *_date, *_time, *_status, *_flag, *_code, *_key, *_count, *_valid, *_verified, *_updated), choose 'Unsure'.
            3) Treat "Column Length" as the maximum number of characters stored in this field and use it to check plausibility:
            - Age: 1–3 digits
            - Phone: 7–15 digits (may include separators)
            - Postal Code: 4–10
            - SSN (National Identifier): 9 digits
            - Medicare ID: ~10–11
            - Names (First/Last/Full): typically ≤ 255
            - Address/Address 2: free text, often 255
            - Dates: 8–10 characters if stored as text (e.g., YYYY-MM-DD)
            If the length strongly contradicts a candidate label, prefer 'Unsure'.
            4) Prefer the most specific match (e.g., 'Address 2' over 'Address' if line-2/apt/suite is implied).
            5) When torn between multiple labels, pick the one supported by the most exact keyword match.
            6) Tokens like 'name', 'fname', 'lname', 'patient', 'customer' refer to a PERSON and never imply Address.
            7) Only output 'Address' or 'Address 2' when tokens explicitly include one of: address, addr, street, road, lane, avenue, suite, apt, apartment, flat.
            """
            "## Allowed labels\n"
            f"{labels_csv}\n\n"

            "## PII Reference\n"
            f"{guide_text}\n\n"

            "## Token Glossary (for this column)\n"
            f"{glossary}\n\n"

            "## Few-shot guidance\n"
            "Example A\n"
            "  Column (raw): Name\n"
            "  Normalized: name | Tokens: [name]\n"
            "  Output:\n"
            "    Classification: Full name\n"
            "    Reasoning: plain 'name' denotes a person's name\n"
            "Example B\n"
            "  Column (raw): patient_ID\n"
            "  Normalized: patient ID | Tokens: [patient, ID]\n"
            "  Output:\n"
            "    Classification: Medical Records\n"
            "    Reasoning: 'patient' + 'ID' denotes a patient's medical records; never an address\n"
            "Example C\n"
            "  Column (raw): fname\n"
            "  Normalized: fname | Tokens: [fname]\n"
            "  Output:\n"
            "    Classification: First name\n"
            "    Reasoning: 'fname' means first name\n"
            "Example D\n"
            "  Column (raw): lname\n"
            "  Normalized: lname | Tokens: [lname]\n"
            "  Output:\n"
            "    Classification: Last name\n"
            "    Reasoning: 'lname' means last name\n"
            "Example E\n"
            "  Column (raw): customer_phone\n"
            "  Normalized: customer phone | Tokens: [customer, phone]\n"
            "  Output:\n"
            "    Classification: Phone\n"
            "    Reasoning: contains 'phone'\n"
            "Example F\n"
            "  Column (raw): inp_PDOB\n"
            "  Normalized: pdob | Tokens: [pdob]\n"
            "  Output:\n"
            "    Classification: Date of birth\n"
            "    Reasoning: 'pdob' → Date of birth\n"
            "Example G\n"
            "  Column (raw): inp_PFN\n"
            "  Normalized: pfn | Tokens: [pfn]\n"
            "  Output:\n"
            "    Classification: First name\n"
            "    Reasoning: 'pfn' → first name\n"
            "Example H\n"
            "  Column (raw): inp_PLN\n"
            "  Normalized: pln | Tokens: [pln]\n"
            "  Output:\n"
            "    Classification: Last name\n"
            "    Reasoning: 'pln' → last name\n\n"

            "## Column Context\n"
            f"- Column name (raw): {column_name}\n"
            f"- Column name (normalized): {normalized}\n"
            f"- Tokens: {tokens}\n"
            f"- Datatype: {dtype_text}\n"
            f"- Column Length: {col_length}\n\n"

            "Respond in EXACTLY two lines:\n"
            "Classification: <one Allowed label OR Unsure>\n"
            "Reasoning: <few words, concise>"
        )



        self._log(f"[REQUEST] provider={self.cfg.provider} model={self.cfg.model} url={self.cfg.base_url}")
        self._log(f"[CONTEXT] raw='{column_name}' | norm='{normalized}' | tokens={tokens} | dtype='{dtype_text}' | length='{col_length}'")
        self._log(f"[ALLOWED] {labels_csv}")
        self._log(f"[PROMPT_PREVIEW]\n{prompt[:1200]}{'...<trimmed>' if len(prompt)>1200 else ''}")

        # ---- call model based on provider ----
        try:
            if self.cfg.provider == "ollama":
                return self._call_ollama(prompt, label_space)
            elif self.cfg.provider == "openai":
                return self._call_openai(prompt, label_space)
            else:
                logger.error(f"Unknown provider: {self.cfg.provider}")
                return None, ""
        except Exception as e:
            logger.error(f"[ERROR] model call failed: {e}")
            return None, ""

    def _call_ollama(self, prompt: str, label_space: List[str]) -> Tuple[Optional[str], str]:
        url = f"{self.cfg.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "top_p": 1, "num_predict": 128}
        }
        try:
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Connection": "close"},
                              timeout=self.cfg.request_timeout)
            if r.status_code == 404:
                return self._call_ollama_chat(prompt, label_space)
            r.raise_for_status()
            data = r.json()
            text = (data.get("response") or "").strip()
            return self._canonicalize_response(text, label_space)
        except requests.exceptions.RequestException:
            return self._call_ollama_chat(prompt, label_space)

    def _call_ollama_chat(self, prompt: str, label_space: List[str]) -> Tuple[Optional[str], str]:
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "top_p": 1, "num_predict": 128}
        }
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json", "Connection": "close"},
                          timeout=self.cfg.request_timeout)
        r.raise_for_status()
        data = r.json()
        if "message" in data and "content" in data["message"]:
            text = (data["message"]["content"] or "").strip()
        else:
            text = (data.get("response") or "").strip()
        return self._canonicalize_response(text, label_space)

    def _call_openai(self, prompt: str, label_space: List[str]) -> Tuple[Optional[str], str]:
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 128
        }
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                          timeout=self.cfg.request_timeout)
        r.raise_for_status()
        data = r.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        return self._canonicalize_response(text, label_space)

    def _canonicalize_response(self, text: str, label_space: List[str]) -> Tuple[Optional[str], str]:
        """Parse and canonicalize the model response, returning (classification, reasoning)."""
        if not text:
            return None, ""

        cleaned = text.strip()
        classification = ""
        reasoning = ""

        if "Classification:" in cleaned and "Reasoning:" in cleaned:
            lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
            for line in lines:
                if line.lower().startswith("classification:"):
                    classification = line.split(":", 1)[1].strip().strip('"').strip("'")
                elif line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
        else:
            # Flexible extraction
            parts = cleaned.splitlines()
            if parts:
                classification = parts[0].strip()
                if len(parts) > 1:
                    reasoning = " ".join(p.strip() for p in parts[1:])

        classification = classification[:200]
        reasoning = reasoning[:200]

        mapped = _canonicalize(classification, label_space)
        self._log(f"[PARSE] '{classification}' -> mapped={mapped}; reason='{reasoning}'")
        return mapped, reasoning

# --- Validator (unchanged 3-column output) -------------------

class TargetedPIIValidator:
    def __init__(self, cfg: ModelConfig):
        self.client = ModelClient(cfg)
        self.label_space = cfg.label_space or list(PII_LABELS)

    def validate_row(
        self,
        column_name: str,
        guessed: str,
        datatype: Optional[str] = None,
        col_length: Optional[Union[str, int]] = None
    ) -> Tuple[str, str, str]:
        """Returns (slm_guess, slm_confidence, slm_reasoning)."""
        mapped, model_reasoning = self.client.query(column_name, datatype, col_length, self.label_space)

        if not mapped:
            slm_guess = ""
            slm_confidence = "Unsure"
            slm_reasoning = model_reasoning if model_reasoning else "Model could not classify column"
            logger.info(f"[VERDICT] column='{column_name}' -> Unsure (expected='{(guessed or '').strip()}')")
            return slm_guess, slm_confidence, slm_reasoning

        slm_guess = mapped
        if mapped.strip().lower() == (guessed or "").strip().lower():
            slm_confidence = "True Positive"
            slm_reasoning = ""  # per your original behavior
            logger.info(f"[VERDICT] column='{column_name}' -> True Positive")
        else:
            slm_confidence = "Negative"
            slm_reasoning = model_reasoning or (
                f"Model classified as '{mapped}' but expected '{(guessed or '').strip() or '—'}'"
            )
            logger.info(f"[VERDICT] column='{column_name}' -> Negative: {mapped}")

        return slm_guess, slm_confidence, slm_reasoning

