import re
import json
import logging
import requests
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import Counter

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
]

# --- Column name normalization for prompt clarity ------------

_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])')
_NON_ALNUM_RE = re.compile(r'[^a-z0-9]+')

def _normalize_colname(name: str) -> str:
    if not name:
        return ""
    s = str(name)
    
    # Handle bracketed database-style column names like [inp_PSSN], [pd_SFName]
    s = re.sub(r'[\[\]]', '', s)  # Remove brackets
    
    # Handle common database prefixes but preserve the main part
    s = re.sub(r'^(inp_|pd_|tbl_|usr_|cust_)', '', s)  # Remove common prefixes
    
    # Handle camelCase
    s = _CAMEL_RE.sub(" ", s)
    
    # Replace underscores and non-alphanumeric with spaces  
    s = re.sub(r'[_\W]+', ' ', s)
    
    # Clean up and return
    return " ".join(s.split()).lower()

def _analyze_all_words(column_name: str, label_space: List[str]) -> Optional[str]:
    """
    Analyze ALL identifiable words in column name and return PII type only if there's consensus.
    Returns None if multiple conflicting PII types are detected.
    """
    if not column_name:
        return None
        
    # Normalize and extract all words
    normalized = _normalize_colname(column_name)
    words = normalized.split()
    
    # Word-to-PII-type mapping
    word_mappings = {
        # Medicare/Insurance (HIGH PRIORITY - check context)
        "medicare": ["Medicare ID"],
        "hic": ["Medicare ID"], 
        "mco": ["Medicare ID"],  # MCO = Managed Care Organization (Medicare context)
        "insurance": ["Insurance number"],
        "policy": ["Insurance number"],
        
        # National IDs
        "ssn": ["National Identifier"],
        "pssn": ["National Identifier"], 
        "sssn": ["National Identifier"],
        "dssn": ["National Identifier"],
        "social": ["National Identifier"],
        "security": ["National Identifier"],
        "national": ["National Identifier"],
        "aadhaar": ["National Identifier"],
        "aadhar": ["National Identifier"],
        "pan": ["National Identifier"],
        
        # Names (can be ambiguous)
        "fname": ["First name"],
        "sfname": ["First name"],
        "dfname": ["First name"], 
        "first": ["First name"],
        "given": ["First name"],
        "firstname": ["First name"],
        
        "lname": ["Last name"],
        "slname": ["Last name"],
        "dlname": ["Last name"],
        "dmname": ["Last name"],
        "last": ["Last name"],
        "surname": ["Last name"],
        "family": ["Last name"],
        "lastname": ["Last name"],
        
        "full": ["Full name"],
        "complete": ["Full name"],
        "employee": ["Full name"],  # employee_name -> Full name
        
        # Contact Info (context-dependent)
        "email": ["Email addresses"],
        "mail": ["Email addresses"],
        
        "phone": ["Phone"],
        "mobile": ["Phone"],
        "cell": ["Phone"],
        "telephone": ["Phone"],
        "tel": ["Phone"],
        
        # Address
        "address": ["Address"],
        "addr": ["Address"],
        "street": ["Address"], 
        "home": ["Address"],
        "saddress": ["Address"],
        "daddress": ["Address"],
        
        # Location
        "city": ["City"],
        "town": ["City"],
        "country": ["Country"],
        
        # Postal
        "zip": ["Postal Code"],
        "postal": ["Postal Code"],
        "pin": ["Postal Code"],
        
        # Date/Age
        "dob": ["Date of birth"],
        "birth": ["Date of birth"],
        "ddob": ["Date of birth"],
        "age": ["Age"],
        
        # Other
        "bank": ["Bank account"],
        "account": ["Bank account"],
        "medical": ["Medical records"],
        "license": ["Drivers license number"],
        "licence": ["Drivers license number"],
        "driving": ["Drivers license number"],
        
        # Context words (help disambiguation but don't vote alone)
        "user": [],     # user_email -> email gets the vote
        "customer": [], # customer_phone -> phone gets the vote  
        "number": [],   # mobile_number -> mobile gets the vote
        "code": [],     # zip_code -> zip gets the vote (unless postal context)
        "date": [],     # birth_date -> birth gets the vote
        "name": [],     # too ambiguous alone, needs qualifier
        "id": [],       # too generic alone
        "num": [],      # too generic alone
        "no": [],       # too generic alone
        "plan": [],     # could be many things alone
    }
    
    # SPECIAL CASE: Handle compound patterns that should always win
    compound_patterns = {
        # Medicare compound patterns (highest priority)
        ("hic", "medicare"): "Medicare ID",
        ("mco", "medicare"): "Medicare ID", 
        ("hic", "id", "medicare"): "Medicare ID",
        ("mco", "name", "medicare"): "Medicare ID",
        ("mco", "plan", "medicare"): "Medicare ID",
        
        # Contact compound patterns
        ("user", "email"): "Email addresses",
        ("customer", "phone"): "Phone",
        ("mobile", "number"): "Phone",
        
        # Other compound patterns
        ("zip", "code"): "Postal Code",
        ("birth", "date"): "Date of birth",
        ("employee", "name"): "Full name",
        ("home", "address"): "Address",
    }
    
    # Check for compound patterns first (highest priority)
    words_set = set(words)
    for pattern_words, pii_type in compound_patterns.items():
        if all(word in words_set for word in pattern_words) and pii_type in label_space:
            logger.debug(f"[MULTI_WORD] Compound pattern match: {pattern_words} -> {pii_type}")
            return pii_type
    
    # Collect all potential PII types for each meaningful word
    detected_types = []
    word_contributions = {}
    
    for word in words:
        if word in word_mappings and word_mappings[word]:  # Only count words that vote
            potential_types = word_mappings[word]
            detected_types.extend(potential_types)
            word_contributions[word] = potential_types
    
    logger.debug(f"[MULTI_WORD] '{column_name}' -> normalized: '{normalized}' -> words: {words}")
    logger.debug(f"[MULTI_WORD] word_contributions: {word_contributions}")
    
    if not detected_types:
        logger.debug(f"[MULTI_WORD] No PII patterns detected")
        return None
        
    # Count occurrences of each PII type
    type_counts = Counter(detected_types)
    logger.debug(f"[MULTI_WORD] type_counts: {dict(type_counts)}")
    
    # Get unique PII types that were detected
    unique_types = list(type_counts.keys())
    
    # Filter to only types that are in our label space
    valid_types = [t for t in unique_types if t in label_space]
    
    # Decision logic:
    if len(valid_types) == 0:
        logger.debug(f"[MULTI_WORD] No valid types in label space")
        return None
    elif len(valid_types) == 1:
        # Clear consensus
        result = valid_types[0]
        logger.debug(f"[MULTI_WORD] Clear consensus: {result}")
        return result
    else:
        # Multiple different PII types detected - check for clear winner
        most_common_type, most_common_count = type_counts.most_common(1)[0]
        
        # If one type appears significantly more than others, use it
        if most_common_count > 1 and most_common_type in label_space:
            logger.debug(f"[MULTI_WORD] Clear winner by frequency: {most_common_type} ({most_common_count} votes)")
            return most_common_type
        else:
            # True conflict - multiple types with equal evidence
            logger.debug(f"[MULTI_WORD] True conflict detected: {valid_types} -> returning None for Unsure")
            return None

# --- Canonicalization with multi-word analysis --------------

def _canonicalize(label: str, label_space: list[str]) -> str | None:
    if not label:
        return None

    txt = label.strip()
    if txt.lower() == "unsure":
        return None

    # First try multi-word analysis on the original input
    multi_word_result = _analyze_all_words(txt, label_space)
    if multi_word_result:
        return multi_word_result

    # If no multi-word match, fall back to original synonym-based approach
    norm = txt.lower()

    # Expanded synonym mapping for better pattern recognition
    synonyms = {
        # Medicare variations
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
        
        # Email variations
        "email": "Email addresses",
        "email address": "Email addresses", 
        "emails": "Email addresses",
        "e-mail": "Email addresses",
        "mail": "Email addresses",
        "email addr": "Email addresses",

        # Phone variations  
        "phone number": "Phone",
        "telephone": "Phone",
        "mobile": "Phone", 
        "cell": "Phone",
        "phone no": "Phone",
        "tel": "Phone",
        "mobile number": "Phone",
        "cell phone": "Phone",
        "telephone number": "Phone",

        # Date of birth variations
        "dob": "Date of birth",
        "date of birth": "Date of birth",
        "birthdate": "Date of birth", 
        "birth date": "Date of birth",
        "birth_date": "Date of birth",
        "dateofbirth": "Date of birth",

        # Postal code variations
        "postal code": "Postal Code",
        "postcode": "Postal Code", 
        "zip": "Postal Code",
        "zip code": "Postal Code",
        "zipcode": "Postal Code",
        "zip_code": "Postal Code",
        "pin code": "Postal Code",
        "pincode": "Postal Code",

        # License variations
        "drivers license": "Drivers license number",
        "driver license": "Drivers license number", 
        "driving license": "Drivers license number",
        "license": "Drivers license number",
        "licence": "Drivers license number",

        # Address variations
        "address": "Address",
        "addr": "Address", 
        "street address": "Address",
        "home address": "Address",
        "mailing address": "Address",
        
        "address line 2": "Address 2",
        "addr2": "Address 2",
        "address2": "Address 2",

        # Name variations
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
        "name": "Full name",
        "complete name": "Full name",
        
        # National ID variations (important for SSN patterns)
        "ssn": "National Identifier",
        "social security number": "National Identifier",
        "social security": "National Identifier", 
        "aadhaar": "National Identifier",
        "aadhar": "National Identifier",
        "pan": "National Identifier", 
        "national id": "National Identifier",
        "citizen id": "National Identifier",

        # Other variations
        "age": "Age",
        "city": "City", 
        "town": "City",
        "country": "Country",
        "total":"NOT_PII",
        "amount":"NOT_PII",
        "client":"NOT_PII",
        "complaint":"NOT_PII",
        
        # Bank/financial
        "account": "Bank account",
        "bank account": "Bank account",
        "account number": "Bank account",
        
    }
    
    # Check regular synonyms
    if norm in synonyms and synonyms[norm] in label_space:
        mapped = synonyms[norm]
        logger.debug(f"[MAP] synonym '{txt}' -> '{mapped}'")
        return mapped

    for l in label_space:
        if l.strip().lower() == norm:
            logger.debug(f"[MAP] exact ci-match '{txt}' -> '{l}'")
            return l

    # minimal plural/case fallback
    def _canon_basic(s: str) -> str:
        s2 = s.strip().lower()
        return s2[:-1] if s2.endswith("s") and not s2.endswith("ss") else s2

    canon_txt = _canon_basic(txt)
    for l in label_space:
        if _canon_basic(l) == canon_txt:
            logger.debug(f"[MAP] basic-canon '{txt}' -> '{l}'")
            return l

    logger.debug(f"[MAP] no mapping for '{txt}' -> None")
    return None

# --- Model Client --------------------------------------------

class ModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        if getattr(self.cfg, "debug", False):
            logger.setLevel(logging.DEBUG)

    def _log(self, msg: str):
        if self.cfg.debug:
            logger.debug(msg)

    def query(self, column_name: str, datatype: Optional[str], col_length: Optional[str | int], label_space: List[str]) -> Tuple[Optional[str], str]:
        """Returns (classification, reasoning)"""
        if self.cfg.provider == "none":
            logger.warning("Model provider is 'none' — set --provider ollama/openai and --model <n> to enable SLM.")
            return None, ""

        # Enhanced prompt that asks for brief reasoning
        labels_csv = ", ".join(label_space)
        prompt = (
            f"Column name: '{column_name}'.\n"
            f"Pick one from this list: {labels_csv}.\n"
            f"Format your answer as:\n"
            f"Classification: [LABEL]\n"
            f"Reasoning: [Brief explanation of what in the column name led to this classification]\n\n"
            f"If unsure, just answer 'Unsure'."
        )

        # ---- logging inputs / prompt preview ----
        self._log(f"[REQUEST] provider={self.cfg.provider} model={self.cfg.model} url={self.cfg.base_url}")
        self._log(f"[CONTEXT] raw='{column_name}'")
        self._log(f"[ALLOWED] {labels_csv}")
        self._log(f"[PROMPT_PREVIEW] {prompt}")

        # ---- call model based on provider ----
        try:
            if self.cfg.provider == "ollama":
                return self._call_ollama(prompt)
            elif self.cfg.provider == "openai":
                return self._call_openai(prompt)
            else:
                logger.error(f"Unknown provider: {self.cfg.provider}")
                return None, ""
        except Exception as e:
            logger.error(f"[ERROR] model call failed: {e}")
            return None, ""

    def _call_ollama(self, prompt: str) -> Tuple[Optional[str], str]:
        """Call Ollama API with proper error handling"""
        # Try /api/generate first (most common)
        url = f"{self.cfg.base_url.rstrip('/')}/api/generate"
        self._log(f"[TRY] Ollama generate: {url}")
        
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "num_predict": 128  # Increased for reasoning
            }
        }
        
        try:
            r = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "Connection": "close"},
                timeout=self.cfg.request_timeout,
            )
            
            if r.status_code == 404:
                # Try /api/chat endpoint as fallback
                self._log("[FALLBACK] /api/generate returned 404, trying /api/chat")
                return self._call_ollama_chat(prompt)
            
            r.raise_for_status()
            
            response_data = r.json()
            text = (response_data.get("response") or "").strip()
            
            self._log(f"[RESPONSE] status={r.status_code} len={len(text)}")
            self._log(f"[RAW_OUTPUT] {text}")
            
            return self._canonicalize_response(text)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Ollama generate call failed: {e}")
            # Try chat endpoint as fallback
            return self._call_ollama_chat(prompt)

    def _call_ollama_chat(self, prompt: str) -> Tuple[Optional[str], str]:
        """Fallback to Ollama chat endpoint"""
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        self._log(f"[TRY] Ollama chat: {url}")
        
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "num_predict": 128  # Increased for reasoning
            }
        }
        
        try:
            r = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "Connection": "close"},
                timeout=self.cfg.request_timeout,
            )
            r.raise_for_status()
            
            data = r.json()
            # Handle different response formats
            if "message" in data and "content" in data["message"]:
                text = (data["message"]["content"] or "").strip()
            else:
                text = (data.get("response") or "").strip()
            
            self._log(f"[RESPONSE] chat status={r.status_code} len={len(text)}")
            self._log(f"[RAW_OUTPUT] {text}")
            
            return self._canonicalize_response(text)
            
        except Exception as e:
            logger.error(f"[ERROR] Ollama chat call failed: {e}")
            return None, ""

    def _call_openai(self, prompt: str) -> Tuple[Optional[str], str]:
        """Call OpenAI-compatible API"""
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        self._log(f"[TRY] OpenAI-compatible chat: {url}")
        
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 128  # Increased for reasoning
        }
        
        try:
            r = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.cfg.request_timeout,
            )
            r.raise_for_status()
            
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            
            self._log(f"[RESPONSE] openai status={r.status_code} len={len(text)}")
            self._log(f"[RAW_OUTPUT] {text}")
            
            return self._canonicalize_response(text)
            
        except Exception as e:
            logger.error(f"[ERROR] OpenAI call failed: {e}")
            return None, ""

    def _canonicalize_response(self, text: str) -> Tuple[Optional[str], str]:
        """Parse and canonicalize the model response, returning (classification, reasoning)"""
        if not text:
            return None, ""
        
        # Clean up response - remove extra whitespace
        cleaned = text.strip()
        
        # Check if response has structured format with reasoning
        if "Classification:" in cleaned and "Reasoning:" in cleaned:
            lines = cleaned.split('\n')
            classification_line = ""
            reasoning_line = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Classification:"):
                    classification_line = line.replace("Classification:", "").strip()
                elif line.startswith("Reasoning:"):
                    reasoning_line = line.replace("Reasoning:", "").strip()
            
            # Clean up classification
            classification = classification_line.strip().strip('"').strip("'").strip()
            reasoning = reasoning_line[:200]  # Limit reasoning to 200 chars max
            
            mapped = _canonicalize(classification, self.cfg.label_space or PII_LABELS)
            self._log(f"[PARSE] structured: '{text}' -> classification='{classification}' -> mapped={mapped}, reasoning='{reasoning}'")
            
            return mapped, reasoning
        else:
            # Try to extract reasoning from various formats
            reasoning = ""
            classification_text = cleaned
            
            # Check for common reasoning patterns
            reasoning_patterns = [
                r"(.+?)\s*(?:because|due to|since|as|reason:?)\s*(.+?)(?:\.|$)",
                r"(.+?)\s*-\s*(.+?)(?:\.|$)",
                r"(.+?)\s*\(\s*(.+?)\s*\)",
                r"(.+?)\s*:\s*(.+?)(?:\.|$)",
            ]
            
            for pattern in reasoning_patterns:
                match = re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL)
                if match:
                    potential_classification = match.group(1).strip()
                    potential_reasoning = match.group(2).strip()
                    
                    # Check if first group looks like a valid classification
                    mapped_check = _canonicalize(potential_classification, self.cfg.label_space or PII_LABELS)
                    if mapped_check:
                        classification_text = potential_classification
                        reasoning = potential_reasoning[:200]
                        break
            
            # If no reasoning pattern found, try to extract it from multi-line responses
            if not reasoning and '\n' in cleaned:
                lines = cleaned.split('\n')
                if len(lines) >= 2:
                    classification_text = lines[0].strip()
                    reasoning = ' '.join(lines[1:]).strip()[:200]
            
            # Clean up classification
            classification = classification_text.strip().strip('"').strip("'").strip()
            mapped = _canonicalize(classification, self.cfg.label_space or PII_LABELS)
            
            self._log(f"[PARSE] flexible: '{text}' -> classification='{classification}' -> mapped={mapped}, reasoning='{reasoning}'")
            
            return mapped, reasoning

# --- Validator (enhanced with reasoning) --------------------

class TargetedPIIValidator:
    def __init__(self, cfg: ModelConfig):
        self.client = ModelClient(cfg)
        self.label_space = cfg.label_space or list(PII_LABELS)

    def validate_row(self, column_name: str, guessed: str, datatype: Optional[str] = None, col_length: Optional[str | int] = None) -> Tuple[str, str, str]:
        """Returns (confidence_type, slm_guess, reasoning)"""
        mapped, model_reasoning = self.client.query(column_name, datatype, col_length, self.label_space)

        if not mapped:
            logger.info(f"[VERDICT] column='{column_name}' -> Unsure (guessed='{(guessed or '').strip()}')")
            return "Unsure", "", model_reasoning if model_reasoning else "Model could not classify column"

        if mapped.strip().lower() == (guessed or "").strip().lower():
            confidence_type = "True positive"
            slm_guess = ""
            logger.info(f"[VERDICT] column='{column_name}' -> {confidence_type}")
            return confidence_type, slm_guess, ""  # No reasoning stored for true positives
        else:
            confidence_type = "Negative" 
            slm_guess = mapped
            
            # Generate reasoning for negative cases
            if model_reasoning:
                reasoning = model_reasoning
            else:
                # Generate simple reasoning based on the disagreement
                guessed_clean = (guessed or "").strip()
                if guessed_clean:
                    reasoning = f"Model classified as '{mapped}' but expected '{guessed_clean}'"
                else:
                    reasoning = f"Model classified as '{mapped}' but no classification was expected"
            
            logger.info(f"[VERDICT] column='{column_name}' -> {confidence_type}: {mapped}")
            return confidence_type, slm_guess, reasoning
