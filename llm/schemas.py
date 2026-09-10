from pydantic import BaseModel, Field, model_validator


class CriterionEvaluation(BaseModel):
    criterion_id: str
    criterion_name: str
    score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    evidence: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_score(self):
        if self.score > self.max_score:
            raise ValueError(
                "score cannot exceed max_score"
            )

        return self


class LLMEvaluation(BaseModel):
    criterion_results: list[CriterionEvaluation]
    model: str
    overall_notes: list[str] = Field(default_factory=list)