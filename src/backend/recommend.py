"""Deterministic HR recommendation tiers from a numeric score (AGENTS.md)."""


def get_recommendation(score: int) -> str:
    if score >= 95:
        return "Strong Shortlist"
    elif score >= 85:
        return "Shortlist"
    elif score >= 75:
        return "Interview Recommended"
    elif score >= 65:
        return "Consider"
    elif score >= 55:
        return "Borderline"
    elif score >= 45:
        return "Hold"
    elif score >= 35:
        return "Upskill & Reapply"
    elif score >= 20:
        return "Alternative Role Recommended"
    else:
        return "Reject"
