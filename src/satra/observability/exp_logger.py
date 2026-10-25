import os
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Subtle emphasis for dark terminals: bold cyan text without a full-row background.
MONITOR_HIGHLIGHT = "\033[1;38;5;81m"
PATIENCE_COLORS = [
    "\033[1;38;5;45m",
    "\033[1;38;5;39m",
    "\033[1;38;5;220m",
    "\033[1;38;5;208m",
    "\033[1;38;5;196m",
]
ANSI_RESET = "\033[0m"


class ExpLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")

    def _color_enabled(self):
        return True

    def _patience_color(self, patience_ratio):
        if patience_ratio is None:
            return None
        ratio = max(0.0, min(1.0, float(patience_ratio)))
        index = min(int(ratio * len(PATIENCE_COLORS)), len(PATIENCE_COLORS) - 1)
        return PATIENCE_COLORS[index]

    def _colorize_patience_cell(self, text, patience_ratio):
        color = self._patience_color(patience_ratio)
        if color is None or "patience" not in text:
            return text
        start = text.find("patience")
        end = text.find(" |", start)
        if end == -1:
            end = len(text)
        return text[:start] + color + text[start:end] + MONITOR_HIGHLIGHT + text[end:]

    def log(self, message="", highlight=False, patience_ratio=None):
        text = str(message)
        terminal_text = text
        if highlight and self._color_enabled():
            terminal_text = self._colorize_patience_cell(text, patience_ratio)
            terminal_text = f"{MONITOR_HIGHLIGHT}{terminal_text}{ANSI_RESET}"
        sys.stdout.write(terminal_text + "\n")
        sys.stdout.flush()
        self._fh.write(ANSI_RE.sub("", text) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()
