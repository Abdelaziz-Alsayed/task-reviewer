from dataclasses import asdict
from .deterministic import run_checks


def evaluate(extracted: dict, rubric: dict) -> dict:
    checks = run_checks(extracted)
    criterion_results = []
    total = 0.0
    for criterion in rubric["criteria"]:
        score = 0.0
        evidence = []
        for item in criterion["checks"]:
            for check_id in item["checks"]:
                result = checks[check_id]
                if result.passed:
                    score += item["points"]
                evidence.append({
                    "check_id": check_id,
                    "passed": result.passed,
                    "evidence": result.evidence,
                    "confidence": result.confidence,
                })
        score = min(score, criterion["max_score"])
        total += score
        criterion_results.append({
            "criterion_id": criterion["id"],
            "criterion": criterion["name"],
            "max_score": criterion["max_score"],
            "score": round(score, 2),
            "evidence": evidence,
        })
    return {
        "score": round(total, 2),
        "max_score": rubric["assignment"]["total_marks"],
        "criteria": criterion_results,
        "checks": {k: asdict(v) for k, v in checks.items()},
    }
