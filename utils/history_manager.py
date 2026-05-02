# utils/history_manager.py
# Tracks all previously processed PDFs with metadata.

import os
import json
from datetime import datetime

HISTORY_FILE = os.path.join("data", "upload_history.json")


def load_history() -> list:
    """
    Loads upload history from JSON file.
    Returns empty list if no history exists yet.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: list):
    """
    Saves upload history list to JSON file.
    Creates data/ folder if it doesn't exist.
    """
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_to_history(
    filename: str,
    chunks: int,
    file_size_bytes: int
):
    """
    Adds a new entry to upload history.
    Called every time a PDF is successfully processed.
    """
    history = load_history()

    # Check if file already exists in history — update it
    for entry in history:
        if entry["filename"] == filename:
            entry["chunks"]        = chunks
            entry["file_size_kb"]  = round(file_size_bytes / 1024, 1)
            entry["last_processed"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            )
            save_history(history)
            return

    # New entry
    history.append({
        "filename":       filename,
        "chunks":         chunks,
        "file_size_kb":   round(file_size_bytes / 1024, 1),
        "first_uploaded": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_processed": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    save_history(history)


def remove_from_history(filename: str):
    """
    Removes a specific entry from upload history.
    """
    history = load_history()
    history = [e for e in history if e["filename"] != filename]
    save_history(history)


def clear_history():
    """
    Wipes the entire upload history.
    """
    save_history([])


def get_history_stats() -> dict:
    """
    Returns summary stats about all processed documents.
    """
    history = load_history()
    if not history:
        return {
            "total_docs":   0,
            "total_chunks": 0,
            "total_size_kb": 0.0
        }

    return {
        "total_docs":    len(history),
        "total_chunks":  sum(e["chunks"] for e in history),
        "total_size_kb": round(
            sum(e["file_size_kb"] for e in history), 1
        )
    }