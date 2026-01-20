import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import google.generativeai as genai


def _normalize_stop_sequences(stop):
    """Gemini allows at most 5 stop sequences. Accept str or list/tuple."""
    if stop is None:
        return None
    if isinstance(stop, str):
        stops = [stop]
    elif isinstance(stop, (list, tuple)):
        stops = list(stop)
    else:
        stops = [str(stop)]

    # Drop empty + de-duplicate preserving order
    seen = set()
    out = []
    for s in stops:
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)

    return out[:5]

def _get_gemini_api_key() -> str:
    # Prefer a single env var, but accept common alternatives.
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        or ""
    )


def _chatml_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert ChatML-style messages to a single prompt string."""
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip().upper()
        content = (m.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts).strip()


def query_llm(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    stop: Optional[List[str]] = None,
    return_usage: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]], None]:
    """Gemini-only LLM query helper.

    Keeps the same signature used by evaluation/auto_scorer.py.

    If callers pass non-Gemini model names (e.g., GPT), we ignore them and fall back to:
      - EVAL_GEMINI_MODEL env var, else
      - models/gemini-2.0-flash
    """

    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
        )

    genai.configure(api_key=api_key)

    chosen_model = model
    if not ("gemini" in (model or "").lower() or (model or "").startswith("models/")):
        chosen_model = os.environ.get("EVAL_GEMINI_MODEL", "models/gemini-2.0-flash")

    prompt = _chatml_to_prompt(messages)

    last_err: Optional[BaseException] = None
    for _ in range(int(os.environ.get("LLM_API_RETRIES", "2"))):
        try:
            gm = genai.GenerativeModel(chosen_model)
            generation_config: Dict[str, Any] = {
                "temperature": float(temperature),
                "max_output_tokens": int(max_new_tokens),
            }
            # Always normalize any stop_sequences shape to Gemini's constraints.
            if stop is not None:
                generation_config["stop_sequences"] = _normalize_stop_sequences(stop)
            if "stop_sequences" in generation_config:
                generation_config["stop_sequences"] = _normalize_stop_sequences(
                    generation_config.get("stop_sequences")
                )

            timeout_s = float(os.environ.get("LLM_API_TIMEOUT_S", "60"))
            try:
                resp = gm.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": timeout_s},
                )
            except TypeError:
                # Older versions of google-generativeai may not accept request_options.
                resp = gm.generate_content(prompt, generation_config=generation_config)
            # resp.text can throw if no valid parts (finish_reason=2 etc.)
            try:
                text = resp.text
            except Exception as e:
                raise RuntimeError(
                    "Gemini response has no text (empty/blocked output). "
                    f"Original error: {e}"
                )

            text = (text or "").strip()

            if return_usage:
                usage: Dict[str, Any] = {}
                try:
                    um = getattr(resp, "usage_metadata", None)
                    if um is not None:
                        usage = {
                            "prompt_tokens": getattr(um, "prompt_token_count", None),
                            "completion_tokens": getattr(um, "candidates_token_count", None),
                            "total_tokens": getattr(um, "total_token_count", None),
                        }
                except Exception:
                    usage = {}
                return text, usage

            return text

        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            time.sleep(float(os.environ.get("LLM_API_RETRY_SLEEP_S", "1")))
    if last_err:
        traceback.print_exc()
    return None


if __name__ == "__main__":
    msg = [{"role": "user", "content": "Say hello in one short sentence."}]
    out = query_llm(
        msg,
        model=os.environ.get("EVAL_GEMINI_MODEL", "models/gemini-2.0-flash"),
        temperature=0.0,
        max_new_tokens=32,
    )
    print(out)