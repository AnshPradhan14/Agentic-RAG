"""
markdown_cleaner.py — Backward-compatibility shim

All cleaning logic has moved to src.ingestion.utils_cleaning.
This module re-exports the MarkdownCleaner class so existing call sites
(e.g., legacy code or tests) continue to work without changes.

New code should import directly from utils_cleaning.
"""

import re
import html
import logging
from src.ingestion.utils_cleaning import (
    clean_bilingual_text,
    clean_text_block,
    clean_json_recursive,
    fix_spaced_characters,
    remove_devanagari,
)

logger = logging.getLogger(__name__)


class MarkdownCleaner:
    """Backward-compatible wrapper around utils_cleaning functions."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Full cleaning pipeline for markdown text.
        Delegates to utils_cleaning.clean_text_block.
        """
        return clean_text_block(text)

    @staticmethod
    def fix_markdown_tables(text: str) -> str:
        """Ensure every markdown table has a valid separator row."""
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            line_str = line.strip()
            is_table_line = line_str.startswith('|') and line_str.count('|') >= 2

            if is_table_line:
                table_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    next_str = next_line.strip()
                    if next_str.startswith('|') and next_str.count('|') >= 2:
                        table_lines.append(next_line)
                        i += 1
                    else:
                        break

                header = table_lines[0].strip()
                if not header.endswith('|'):
                    header += '|'
                    table_lines[0] = header

                num_cols = header.count('|') - 1
                new_sep = '|' + '|'.join(['---'] * num_cols) + '|'

                if len(table_lines) > 1:
                    second_line = table_lines[1].strip()
                    is_sep = bool(re.match(r'^[\s\|\-\:]+$', second_line)) and '-' in second_line
                    if is_sep:
                        table_lines[1] = new_sep
                    else:
                        table_lines.insert(1, new_sep)
                else:
                    table_lines.append(new_sep)

                for j in range(2, len(table_lines)):
                    row = table_lines[j].strip()
                    if row != new_sep and not row.endswith('|'):
                        table_lines[j] = table_lines[j] + '|'

                fixed_lines.extend(table_lines)
            else:
                fixed_lines.append(line)
                i += 1

        return '\n'.join(fixed_lines)

    @classmethod
    def clean_json(cls, obj):
        """Recursively apply clean_text to all strings in a JSON-like structure."""
        return clean_json_recursive(obj)
