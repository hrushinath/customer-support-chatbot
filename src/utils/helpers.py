"""
Helper utilities
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib


def save_json(data: Any, file_path: Path, pretty: bool = True) -> None:
    """Save data to JSON file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)


def load_json(file_path: Path) -> Any:
    """Load data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_file_hash(file_path: Path) -> str:
    """Get SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def format_sources(sources: List[Dict]) -> str:
    """Format source information for display"""
    if not sources:
        return "No sources"
    
    lines = []
    for source in sources:
        line = f"- {source.get('source', 'unknown')}"
        if 'similarity' in source:
            line += f" (similarity: {source['similarity']:.2f})"
        lines.append(line)
    
    return "\n".join(lines)
