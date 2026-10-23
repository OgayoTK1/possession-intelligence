"""Schema-driven reader for Metrica's FIFA/EPTS tracking export (metadata XML + raw text).

Why not the official baseline's parser: it assumes the raw line lists slots in the order the
<Player> elements appear, and that every group is "x,y,speed". The EPTS standard says otherwise --
the column layout is defined by <DataFormatSpecification>, which can differ per frame range
(substitutions) and can order players differently from <Players> (Metrica's own open Sample Game 3
does exactly that). This reader follows the specification, records what it found in a schema
report, and refuses to guess when the file disagrees with it.

Output is a `Match` in metres, NaN for missing, indexed by row = frame_id - first_frame_id.
"""

import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .match import Match

CHUNK_LINES = 20000


def _strip_ns(root):
    for element in root.iter():
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]
    return root


def _text(element, path, default=None):
    found = element.find(path)
    return found.text if found is not None and found.text is not None else default


def parse_metadata(meta_path):
    """Teams, players, field size, and the per-frame-range column layout.

    Returns a dict:
      teams: [(team_id, name)] in file order (first = home, FIFA/EPTS convention)
      players: {player_id: team_id}
      channel_of: {player_channel_id: (player_id, channel_id)}
      field: (length, width) from <FieldSize>, or None
      specs: [{start, stop, layout}] where layout is a list of groups; each group is a list of
             ("player", player_id, channel) or ("ball", None, channel) or ("frame", None, None)
    """
    root = _strip_ns(ET.parse(meta_path).getroot())

    teams = [(team.get("id"), _text(team, "Name", "")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    players = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_order = [p.get("id") for p in root.iter("Player")]

    channel_of = {}
    for ch in root.iter("PlayerChannel"):
        channel_of[ch.get("id")] = (ch.get("playerId"), ch.get("channelId"))

    field_size = None
    fs = root.find(".//FieldSize")
    if fs is not None:
        try:
            field_size = (float(_text(fs, "Width")), float(_text(fs, "Height")))
        except (TypeError, ValueError):
            field_size = None

    frame_rate = _text(root, ".//GlobalConfig/FrameRate")

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        start = int(spec.get("startFrame"))
        stop = int(spec.get("endFrame")) + 1
        outer_sep = spec.get("separator", ":")
        groups = []
        for child in spec:
            if child.tag == "StringRegister":
                groups.append({"kind": "frame", "fields": [("frame", None, child.get("name"))]})
            elif child.tag == "SplitRegister":
                groups.append(_split_register(child, channel_of))
        specs.append({"start": start, "stop": stop, "outer_sep": outer_sep, "groups": groups})
    if not specs:
        raise ValueError("{}: no DataFormatSpecification".format(meta_path))
    specs.sort(key=lambda s: s["start"])

    return {"teams": teams, "players": players, "player_order": player_order,
            "channel_of": channel_of, "field": field_size, "specs": specs,
            "frame_rate": int(round(float(frame_rate))) if frame_rate else None}


def _split_register(element, channel_of):
    """A top-level SplitRegister: either the ';'-separated players block or the ball block."""
    sep = element.get("separator", ",")
    nested = [c for c in element if c.tag == "SplitRegister"]
    if nested:
        # players block: each nested register is one player's channels
        players = []
        for reg in nested:
            fields = []
            for ref in reg:
                pid, ch = channel_of[ref.get("playerChannelId")]
                fields.append(("player", pid, ch))
            players.append({"sep": reg.get("separator", ","), "fields": fields})
        return {"kind": "players", "sep": sep, "players": players}
    fields = []
    for ref in element:
        if ref.tag == "BallChannelRef":
            fields.append(("ball", None, ref.get("channelId")))
        elif ref.tag == "PlayerChannelRef":
            pid, ch = channel_of[ref.get("playerChannelId")]
            fields.append(("player", pid, ch))
        else:
            fields.append(("other", None, ref.get("channelId") or ref.tag))
    return {"kind": "ball" if any(f[0] == "ball" for f in fields) else "other",
            "sep": sep, "fields": fields}


def _flat_columns(spec):
    """Column names, in raw-line order, after every separator is replaced by one comma."""
    cols = []
    for group in spec["groups"]:
        if group["kind"] == "frame":
            cols.append("frame")
        elif group["kind"] == "players":
            for player in group["players"]:
                for _, pid, ch in player["fields"]:
                    cols.append("p|{}|{}".format(pid, ch))
        else:
            for kind, pid, ch in group["fields"]:
                cols.append("b|{}".format(ch) if kind == "ball" else "o|{}".format(ch))
    return cols


def _read_block(lines, cols, seps):
    table = str.maketrans({s: "," for s in seps if s != ","})
    text = "".join(line.translate(table) for line in lines)
    frame = pd.read_csv(io.StringIO(text), header=None, names=cols, na_values=["NaN", "nan", ""],
                        dtype=np.float64, engine="c")
    return frame


def load_epts(meta_path, raw_path, pitch=None, name="match", n_frames=None):
    """Read a whole EPTS export into a Match (row i = frame_id - 1 when ids start at 1).

    pitch: (length, width) metres used to de-normalise. Defaults to the metadata FieldSize.
    n_frames: if given, the result has exactly this many rows and any frame_id outside is an error.
    Returns (match, schema_report).
    """
    meta = parse_metadata(meta_path)
    if pitch is None:
        if meta["field"] is None:
            raise ValueError("no pitch size given and none in the metadata")
        pitch = meta["field"]
    length, width = float(pitch[0]), float(pitch[1])

    home_id = meta["teams"][0][0]
    # Global slot order: the <Players> order, so slot k is "Track k+1" in the competition file.
    slot_ids = list(meta["player_order"])
    slot_index = {pid: k for k, pid in enumerate(slot_ids)}
    team_of_slot = np.array([0 if meta["players"][pid] == home_id else 1 for pid in slot_ids],
                            dtype=np.int8)

    specs = meta["specs"]
    spec_cols = [_flat_columns(s) for s in specs]
    spec_seps = [{s["outer_sep"], ";", ","} | _all_seps(s) for s in specs]

    frames_seen, chunks = [], []
    buffer, buffer_spec = [], None
    report = {"n_lines": 0, "n_bad_lines": 0, "specs": [], "ball_channels": None,
              "player_channels": None, "player_order_matches_layout": None}

    def flush():
        if buffer:
            chunks.append((buffer_spec, _read_block(buffer, spec_cols[buffer_spec],
                                                    spec_seps[buffer_spec])))

    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            head = line[:line.index(":")] if ":" in line else None
            try:
                fid = int(head)
            except (TypeError, ValueError):
                report["n_bad_lines"] += 1
                continue
            k = _spec_for(specs, fid)
            if k is None:
                report["n_bad_lines"] += 1
                continue
            if k != buffer_spec or len(buffer) >= CHUNK_LINES:
                flush()
                buffer, buffer_spec = [], k
            buffer.append(line if line.endswith("\n") else line + "\n")
            report["n_lines"] += 1
    flush()

    first_id = min(int(c["frame"].iloc[0]) for _, c in chunks)
    last_id = max(int(c["frame"].iloc[-1]) for _, c in chunks)
    t = n_frames if n_frames is not None else last_id - first_id + 1
    n_slots = len(slot_ids)
    px = np.full((t, n_slots), np.nan, dtype=np.float32)
    py = np.full((t, n_slots), np.nan, dtype=np.float32)
    ball = np.full((t, 2), np.nan, dtype=np.float32)
    extras = {}

    for k, chunk in chunks:
        rows = chunk["frame"].values.astype(np.int64) - first_id
        if rows.min() < 0 or rows.max() >= t:
            raise ValueError("frame ids outside the expected {} rows".format(t))
        frames_seen.append(rows)
        for col in chunk.columns:
            if col.startswith("p|"):
                _, pid, ch = col.split("|")
                s = slot_index[pid]
                if ch in ("x",):
                    px[rows, s] = chunk[col].values * length
                elif ch in ("y",):
                    py[rows, s] = chunk[col].values * width
                else:
                    key = "player_" + ch
                    if key not in extras:
                        extras[key] = np.full((t, n_slots), np.nan, dtype=np.float32)
                    extras[key][rows, s] = chunk[col].values
            elif col.startswith("b|"):
                ch = col[2:]
                if ch == "x":
                    ball[rows, 0] = chunk[col].values * length
                elif ch == "y":
                    ball[rows, 1] = chunk[col].values * width
                else:
                    key = "ball_" + ch
                    if key not in extras:
                        extras[key] = np.full(t, np.nan, dtype=np.float32)
                    extras[key][rows] = chunk[col].values

    seen = np.concatenate(frames_seen)
    report.update({
        "first_frame_id": first_id, "last_frame_id": last_id, "n_rows": t,
        "contiguous": bool(len(np.unique(seen)) == t and len(seen) == t),
        "duplicate_rows": int(len(seen) - len(np.unique(seen))),
        "teams": meta["teams"], "home_team_id": home_id, "n_slots": n_slots,
        "n_home_slots": int(np.sum(team_of_slot == 0)),
        "ball_channels": sorted({c[2:] for _, ch in chunks for c in ch.columns if c.startswith("b|")}),
        "player_channels": sorted({c.split("|")[2] for _, ch in chunks for c in ch.columns
                                   if c.startswith("p|")}),
        "other_channels": sorted({c[2:] for _, ch in chunks for c in ch.columns if c.startswith("o|")}),
        "specs": [(s["start"], s["stop"] - 1, len(c)) for s, c in zip(specs, spec_cols)],
        "player_order_matches_layout": _order_matches(specs, slot_ids),
        "pitch_used_m": (length, width), "field_in_metadata": meta["field"],
        "frame_rate": meta["frame_rate"],
        "ball_present_frac": float(np.mean(np.isfinite(ball[:, 0]))),
        "mean_tracked_players": float(np.mean(np.sum(np.isfinite(px), axis=1))),
    })

    match = Match(name=name, fps=meta["frame_rate"] or 25, pitch=(length, width), ball=ball,
                  px=px, py=py, team_of_slot=team_of_slot, blocks=[(0, t)], extras=extras)
    match.extras["first_frame_id"] = first_id
    return match.validate(), report


def _all_seps(spec):
    seps = set()
    for group in spec["groups"]:
        seps.add(group.get("sep", ","))
        for player in group.get("players", []):
            seps.add(player["sep"])
    return seps


def _spec_for(specs, fid):
    for k, spec in enumerate(specs):
        if spec["start"] <= fid < spec["stop"]:
            return k
    return None


def _order_matches(specs, slot_ids):
    """Does the raw layout list players in <Players> order? (The official baseline assumes so.)"""
    results = []
    for spec in specs:
        order = [p["fields"][0][1] for g in spec["groups"] if g["kind"] == "players"
                 for p in g["players"]]
        results.append(order == [pid for pid in slot_ids if pid in set(order)])
    return all(results)
