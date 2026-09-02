from pydantic import BaseModel, Field


class CriterionEvaluation(BaseModel):
    """LLM evaluation for one rubric criterion."""

    criterion_id: str
    criterion_name: str

    score: float = Field(ge=0)

    max_score: float = Field(gt=0)

    evidence: list[str] = Field(default_factory=list)

    missing_requirements: list[str] = Field(default_factory=list)

    reasoning: str

    confidence: float = Field(ge=0, le=1)


class LLMEvaluation(BaseModel):
    """Complete criterion-level LLM evaluation."""

    criterion_results: list[CriterionEvaluation]

    model: str

    overall_notes: list[str] = Field(default_factory=list)