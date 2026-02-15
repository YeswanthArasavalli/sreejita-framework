# sreejita/narrative/engine.py

from dataclasses import dataclass
from typing import Dict, Any, List


# =====================================================
# OUTPUT MODELS (STABLE, NON-INTERPRETIVE)
# =====================================================

@dataclass
class ActionItem:
    action: str
    owner: str
    timeline: str
    expected_outcome: str


@dataclass
class NarrativeResult:
    executive_summary: List[str]
    key_insights: List[str]
    risks: List[str]
    action_plan: List[ActionItem]
    policy_notes: List[str]


# =====================================================
# NARRATIVE ENGINE (FORMAT-ONLY, GOVERNED)
# =====================================================

def build_narrative(executive_payload: Dict[str, Any]) -> NarrativeResult:
    """
    Narrative Engine — FINAL GOVERNED VERSION

    GUARANTEES:
    - No intelligence computation
    - No re-interpretation of insights
    - No fabricated risks or actions
    - Pure presentation of Executive Cognition output
    """

    if not isinstance(executive_payload, dict):
        return NarrativeResult([], [], [], [], [])

    # -------------------------------------------------
    # EXECUTIVE SUMMARY (PASS-THROUGH)
    # -------------------------------------------------
    summary: List[str] = []

    brief = executive_payload.get("executive_brief")
    if isinstance(brief, str) and brief.strip():
        summary.append(brief.strip())

    board = executive_payload.get("board_readiness")
    if isinstance(board, dict) and board:
        score = board.get("score")
        band = board.get("band")
        if score is not None and band:
            summary.append(
                f"Board Readiness: {score} / 100 ({band})."
            )

    # -------------------------------------------------
    # KEY INSIGHTS (NO REORDERING, NO INFERENCE)
    # -------------------------------------------------
    key_insights: List[str] = []

    insight_block = executive_payload.get("insights", {})
    if isinstance(insight_block, dict):
        for group in ("strengths", "warnings", "risks"):
            for ins in insight_block.get(group, []):
                text = ins.get("so_what") or ins.get("summary")
                if isinstance(text, str) and text.strip():
                    key_insights.append(text.strip())

    # -------------------------------------------------
    # RISKS (ONLY IF EXPLICITLY PROVIDED)
    # -------------------------------------------------
    risks: List[str] = []

    if isinstance(insight_block, dict):
        for ins in insight_block.get("risks", []):
            title = ins.get("title")
            if isinstance(title, str) and title.strip():
                risks.append(title.strip())

    # -------------------------------------------------
    # ACTION PLAN (STRICT PASS-THROUGH)
    # -------------------------------------------------
    actions: List[ActionItem] = []

    for rec in (executive_payload.get("recommendations") or [])[:5]:
        if not isinstance(rec, dict):
            continue

        action = rec.get("action")
        if not action:
            continue

        actions.append(
            ActionItem(
                action=action,
                owner=rec.get("owner", "—"),
                timeline=rec.get("timeline", "—"),
                expected_outcome=rec.get("goal", "—"),
            )
        )

    # -------------------------------------------------
    # POLICY NOTES (VISIBLE, OPTIONAL)
    # -------------------------------------------------
    policy_notes: List[str] = []

    explanations = executive_payload.get("explanations")
    if isinstance(explanations, list):
        for e in explanations:
            if isinstance(e, str) and e.strip():
                policy_notes.append(e.strip())

    # -------------------------------------------------
    # FINAL OUTPUT (NO DEFAULT FABRICATION)
    # -------------------------------------------------
    return NarrativeResult(
        executive_summary=summary[:3],
        key_insights=key_insights[:7],
        risks=risks[:5],
        action_plan=actions,
        policy_notes=policy_notes,
    )


# =====================================================
# BACKWARD-COMPATIBILITY ALIAS
# =====================================================

def generate_narrative(executive_payload: Dict[str, Any]):
    """
    Legacy alias — preserved for backward compatibility.
    """
    return build_narrative(executive_payload)
