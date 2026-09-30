import json


SYSTEM_PROMPT = """
You are an assignment evaluation assistant.

You are evaluating a student's Week 05 Regression assignment.

Your job is NOT to invent a grade.

You must evaluate each rubric criterion using ONLY the supplied:

1. rubric requirements,
2. deterministic evaluation evidence,
3. extracted submission evidence.

You must not assume that something was done if the evidence does not support it.

If evidence is insufficient, explicitly say so.

For each criterion:

- assign a score supported by the evidence,
- provide concise evidence,
- identify missing requirements,
- explain the reasoning,
- provide a confidence value from 0 to 1.

The score must never exceed the criterion's maximum score.

Do not calculate an overall grade.

The application will calculate the total score separately.

The instructor remains the final authority over the grade.

Return ONLY valid JSON.
Do not wrap the JSON in Markdown code fences.
"""


def build_criterion_prompt(evidence: dict) -> str:
    return f"""
Evaluate the following rubric criterion.

RUBRIC AND EVIDENCE
-------------------

{json.dumps(evidence, indent=2, ensure_ascii=False)}

Return ONLY a JSON object using exactly this structure:

{{
  "criterion_id": "string",
  "criterion_name": "string",
  "score": 0,
  "max_score": 0,
  "evidence": [
    "string"
  ],
  "missing_requirements": [
    "string"
  ],
  "reasoning": "string",
  "confidence": 0.0
}}

Important:

- Use only the supplied evidence.
- Do not invent implementation details.
- Do not award marks for requirements that are not supported.
- score must be between 0 and max_score.
- confidence must be between 0 and 1.
- Do not calculate the overall assignment score.
- Return ONLY valid JSON.
"""

def build_full_evaluation_prompt(
        evidence_package: dict,
    ) -> str:
        return f"""
    Evaluate the student's entire assignment using the supplied
    rubric and evidence.
    
    RUBRIC AND EVIDENCE
    -------------------
    
    {json.dumps(evidence_package, indent=2, ensure_ascii=False)}
    
    Evaluate EVERY rubric criterion.
    
    For each criterion:
    
    - assign a score supported by the evidence,
    - provide concise evidence,
    - identify missing requirements,
    - explain the reasoning,
    - provide a confidence value from 0 to 1.
    
    Important rules:
    
    - Use ONLY the supplied rubric and evidence.
    - Do not invent implementation details.
    - Do not assume something was done if the evidence does not support it.
    - Do not award marks for unsupported requirements.
    - The score must be between 0 and the criterion maximum.
    - Use the exact criterion IDs supplied by the rubric.
    - Use the exact maximum score supplied by the rubric.
    - Return one result for EVERY criterion.
    - Do not calculate the overall assignment score.
    - The application will calculate the total score separately.
    - The instructor remains the final authority.
    
    Return ONLY valid JSON.
    
    Use exactly this structure:

    {{
        "criterion_results": [
        {{
            "criterion_id": "string",
            "criterion_name": "string",
            "score": 0,
            "max_score": 0,
            "evidence": [
             "string"
            ],
            "missing_requirements": [
                "string"
            ],
            "reasoning": "string",
            "confidence": 0.0
        }}
    ],
    "model": "string",
    "overall_notes": []
    }}
    """