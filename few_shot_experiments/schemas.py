"""Gemini JSON-mode response schemas for structured output (response_schema param)."""

CONTENT_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "span_text": {"type": "string"},
                },
                "required": ["doc_id", "span_text"],
            },
        }
    },
    "required": ["highlights"],
}

AMBIGUITY_HIGHLIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "span_text": {"type": "string"},
                },
                "required": ["doc_id", "span_text"],
            },
        }
    },
    "required": ["highlights"],
}

FIC_COT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_id": {"type": "integer"},
                    "sentence_text": {"type": "string"},
                    "highlight_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["sentence_id", "sentence_text", "highlight_ids"],
            },
        },
    },
    "required": ["sentences"],
}

SUBTASK_SCHEMAS = {
    "content_selection": CONTENT_SELECTION_SCHEMA,
    "ambiguity_highlight": AMBIGUITY_HIGHLIGHT_SCHEMA,
    "FiC": FIC_COT_SCHEMA,
    "fusion_in_context": FIC_COT_SCHEMA,
}
