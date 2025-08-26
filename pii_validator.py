import re
import requests
from dataclasses import dataclass
from typing import List, Optional

# --- Config ---------------------------------------------------

@dataclass
class ModelConfig:
    provider: str = "none"         # "ollama", "openai", or "none"
    model: str = ""
    base_url: str = "http://localhost:11434"
    label_space: Optional[List[str]] = None

# --- Label canonicalization ----------------------------------

def _canonicalize(label: str, label_space: list[str]) -> str | None:
    if not label:
        return None

    # Normalization
    norm = label.strip().lower()

    # Custom synonym mappings
    synonyms = {
        "zip code": "Postal code",
        "zipcode": "Postal code",
        "postal": "Postal code",
        "address line 1": "Address",
        "addr1": "Address",
        "address line 2": "Address2",
        "addr2": "Address2"
    }

    if norm in synonyms:
        canonical = synonyms[norm]
        if canonical in label_space:
            return canonical

    # Fallback: case-insensitive match against label space
    for l in label_space:
        if l.strip().lower() == norm:
            return l

    return None


# --- Heuristic Guessing --------------------------------------

class HeuristicPIIGuesser:
    def __init__(self, label_space: List[str]):
        self.label_space = label_space

        self.rules = [
            (r"email", "Email address"),
            (r"phone|mobile|fax|tel", "Phone"),
            (r"ssn|pssn|sssn|social", "National Identifier"),
            (r"dob|birth|bdate", "Date of Birth"),
            (r"zip|postal", "Zip code"),
            (r"city", "City"),
            (r"state", "State"),
        ]

        # Non-PII keywords if "NOT_PII" exists in label space
        self.notpii_terms = {"flag", "status", "code", "type", "amount", "value", "id"}

    def guess(self, column_name: str) -> Optional[str]:
        norm = column_name.strip().lower()

        # Non-PII
        if "not_pii" in [l.lower() for l in self.label_space]:
            for t in self.notpii_terms:
                if t in norm:
                    return _canonicalize("NOT_PII", self.label_space)

        # High-precision regex rules
        for pattern, label in self.rules:
            if re.search(pattern, norm):
                return _canonicalize(label, self.label_space)

        # Substring fallbacks for dataset abbreviations
        if "dob" in norm:
            return _canonicalize("Date of Birth", self.label_space)
        if "fname" in norm or "f name" in norm or "sfn" in norm:
            return _canonicalize("First name", self.label_space)
        if "lname" in norm or "l name" in norm or "sln" in norm:
            return _canonicalize("Last name", self.label_space)
        if "address" in norm or "addr" in norm:
            return _canonicalize("Address", self.label_space)

        # Vague "name" → unsure
        if re.search(r"\bname\b", norm):
            return None

        return None

# --- Model Client --------------------------------------------

class ModelClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def query(self, column_name: str, label_space: List[str]) -> Optional[str]:
        # Heuristic first
        h = HeuristicPIIGuesser(label_space).guess(column_name)
        if h:
            return h

        if self.cfg.provider == "none":
            return None

        if self.cfg.provider == "ollama":
            prompt = (
                f"Column name: '{column_name}'.\n"
                f"Pick one from this list: {', '.join(label_space)}.\n"
                f"Answer with only one label or 'Unsure'."
            )
            r = requests.post(
                f"{self.cfg.base_url}/api/generate",
                json={"model": self.cfg.model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            text = r.json().get("response", "").strip()
            return _canonicalize(text, label_space)

        if self.cfg.provider == "openai":
            prompt = (
                f"Column name: '{column_name}'.\n"
                f"Pick one from this list: {', '.join(label_space)}.\n"
                f"Answer with only one label or 'Unsure'."
            )
            r = requests.post(
                f"{self.cfg.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer TOKEN"},
                json={
                    "model": self.cfg.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
                timeout=60,
            )
            text = r.json()["choices"][0]["message"]["content"].strip()
            return _canonicalize(text, label_space)

        return None

# --- Validator ------------------------------------------------

class TargetedPIIValidator:
    def __init__(self, cfg: ModelConfig):
        self.client = ModelClient(cfg)
        self.label_space = cfg.label_space or []

    def validate_row(self, column_name: str, guessed: str) -> str:
        model_guess = self.client.query(column_name, self.label_space)

        if not model_guess:
            return "Unsure"

        if model_guess.lower() == guessed.strip().lower():
            return "True positive"

        return f"Negative: {model_guess}"
