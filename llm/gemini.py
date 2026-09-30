from __future__ import annotations

import json
from typing import Any

from google import genai

from config.llm_settings import GEMINI_API_KEY, GEMINI_MODEL
from llm.base import LLMEvaluator
from llm.prompts import (
    SYSTEM_PROMPT,
    build_criterion_prompt,
    build_full_evaluation_prompt,
)
from llm.schemas import CriterionEvaluation, LLMEvaluation


class GeminiEvaluator(LLMEvaluator):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(api_key=self.api_key)

    def evaluate_criterion(
        self,
        criterion_evidence: dict[str, Any],
    ) -> CriterionEvaluation:
        prompt = build_criterion_prompt(criterion_evidence)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                SYSTEM_PROMPT
                                + "\n\n"
                                + prompt
                            )
                        }
                    ],
                }
            ],
        )

        text = response.text

        if not text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        data = self._parse_json(text)

        expected_id = criterion_evidence["criterion"]["id"]

        if data.get("criterion_id") != expected_id:
            raise ValueError(
                "Gemini returned an unexpected criterion_id: "
                f"{data.get('criterion_id')!r}; "
                f"expected {expected_id!r}"
            )

        return CriterionEvaluation.model_validate(data)

    def evaluate(
        self,
        evidence_package: dict[str, Any],
    ) -> LLMEvaluation:
        prompt = build_full_evaluation_prompt(evidence_package)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                SYSTEM_PROMPT,
                prompt,
            ],
        )

        data = self._parse_json(response.text)
        
        result = LLMEvaluation.model_validate(data)

        return result.model_copy(
            update={"model": self.model}
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON: {exc}"
            ) from exc