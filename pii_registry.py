import json
import os
import string

MAPPING_FILE = "pii_registry.json"


class PIIRegistry:
    """
    Maintains a stable, persisted mapping between real names (PII) and
    redacted tokens, so the LLM never sees real customer names, but the
    same person always maps to the same token across the whole session
    (and across restarts, since the mapping is saved to disk).
    """

    def __init__(self):
        self.to_token = {}   # real name -> token, e.g. {"Alice": "[REDACTED_NAME_A]"}
        self.to_real = {}    # token -> real name, e.g. {"[REDACTED_NAME_A]": "Alice"}
        self._load_mappings()

    def _load_mappings(self):
        if os.path.exists(MAPPING_FILE):
            with open(MAPPING_FILE, "r") as f:
                self.to_token = json.load(f)
            self.to_real = {v: k for k, v in self.to_token.items()}

    def _save_mappings(self):
        with open(MAPPING_FILE, "w") as f:
            json.dump(self.to_token, f, indent=2)

    def redact(self, name: str) -> str:
        if name in self.to_token:
            return self.to_token[name]
        letter = string.ascii_uppercase[len(self.to_token) % 26]
        token = f"[REDACTED_NAME_{letter}]"
        self.to_token[name] = token
        self.to_real[token] = name
        self._save_mappings()
        return token

    def restore(self, token: str) -> str:
        return self.to_real.get(token, token)