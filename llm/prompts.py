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
"""


def build_criterion_prompt(evidence: dict) -> str:
    """Create a prompt for one rubric criterion."""

    return f"""
Evaluate the following rubric criterion.

RUBRIC AND EVIDENCE
-------------------

{json.dumps(evidence, indent=2, ensure_ascii=False)}

Return a criterion-level evaluation that follows the required schema.

Important:
- Use only the supplied evidence.
- Do not invent implementation details.
- Do not award marks for requirements that are not supported.
- Do not calculate the overall assignment score.
"""