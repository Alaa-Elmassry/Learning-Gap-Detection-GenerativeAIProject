import json
import os
import time
import random
import re
from typing import Any, Optional, Tuple

from huggingface_hub import InferenceClient


class HFJSONError(Exception):
    pass


def _extract_json(text: str) -> Any:
    text = (text or "").strip()

    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try find first JSON object/array
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for s in starts:
        cand = text[s:]
        for e in range(len(cand), max(len(cand) - 6000, 0), -1):
            try:
                return json.loads(cand[:e].strip())
            except Exception:
                continue

    raise HFJSONError("Could not parse valid JSON from model output.")


def _looks_like_404(err: Exception) -> bool:
    status = getattr(getattr(err, "response", None), "status_code", None)
    if status == 404:
        return True
    msg = str(err)
    return bool(re.search(r"\b404\b", msg)) and ("Not Found" in msg or "Client Error" in msg)


class HFClient:
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        token_env: str = "HF_TOKEN",
    ):
        token = os.getenv(token_env) or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError(
                f"Missing Hugging Face token. Set env var {token_env} (or HUGGINGFACEHUB_API_TOKEN)."
            )

        provider = os.getenv("HF_PROVIDER", "featherless-ai").strip().lower()
        if provider == "auto":
            provider = "featherless-ai"

        self.model = model
        self.provider = provider
        self.client = InferenceClient(
            model=self.model,
            token=token,
            provider=self.provider,
            timeout=120,
        )

    def generate_json(
        self,
        prompt_text: str,   
        *,
        max_new_tokens: int = 450,
        temperature: float = 0.7,
        top_p: float = 0.9,
        retries: int = 1,
        retry_sleep: float = 1.2,
    ) -> Tuple[Any, str]:
        """
        Returns (parsed_json, raw_text)
        - Try chat_completion first
        - If not supported -> fallback to text_generation
        """
        last_err: Optional[Exception] = None
        system_rule = "You output ONLY valid JSON. No extra text, no markdown, no explanations."

        for attempt in range(1, retries + 1):
            try:
                time.sleep(random.random() * 0.05)

                # 1) Try chat endpoint
                try:
                    resp = self.client.chat_completion(
                        messages=[
                            {"role": "system", "content": system_rule},
                            {"role": "user", "content": prompt_text},
                        ],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    raw = resp.choices[0].message.content

                except Exception as e:
                    # 2) Fallback: plain text generation (works for most providers)
                    full_prompt = f"{system_rule}\n\n{prompt_text}"
                    raw = self.client.text_generation(
                        full_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        return_full_text=False,
                    )

                parsed = _extract_json(raw)
                return parsed, raw

            except Exception as e:
                last_err = e
                time.sleep(retry_sleep * attempt)

        raise HFJSONError(
            f"HF generation failed after {retries} retries. Last error: {last_err}"
        )

