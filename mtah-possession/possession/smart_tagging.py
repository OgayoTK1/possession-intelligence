"""Smart Tagging (SportsCode XML) as frame intervals -- used for the audit only.

It is deliberately NOT a model input in the final system: it exists only for the competition
match, so its effect could be measured on nothing but the 17 public possessions, which the
organisers show is not enough to choose between two versions of a system. See REPORT.md.
"""

import xml.etree.ElementTree as ET
from collections import Counter

FPS = 25


def load_intervals(path):
    """[(start_frame, end_frame, code, {group: text})], frame = round(seconds * 25)."""
    root = ET.parse(path).getroot()
    out = []
    for inst in root.iter("instance"):
        try:
            start = float(inst.findtext("start"))
            end = float(inst.findtext("end"))
        except (TypeError, ValueError):
            continue
        code = (inst.findtext("code") or "").strip()
        tags = {}
        for label in inst.findall("label"):
            group = (label.findtext("group") or "").strip()
            text = (label.findtext("text") or "").strip()
            if group:
                tags[group] = text
        out.append((int(round(start * FPS)), int(round(end * FPS)), code, tags))
    return out


def summary(intervals):
    """Counts only -- nothing frame-level."""
    codes = Counter(code.upper() for _, _, code, _ in intervals)
    teams = Counter(tags.get("Team", "<none>") for _, _, _, tags in intervals)
    types = Counter(tags.get("Type") for _, _, code, tags in intervals
                    if code.upper() == "SET PIECES")
    return {"n_instances": len(intervals), "codes": dict(codes), "team_tags": dict(teams),
            "set_piece_types": {str(k): v for k, v in types.items()}}
