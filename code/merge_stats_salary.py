"""
Merge 2026 MLB player stats with Spotrac cap totals.

Produces two files:
  batters_with_salary.csv   - one row per hitter, stats + salary
  pitchers_with_salary.csv  - one row per pitcher, stats + salary

Usage:
  python merge_stats_salary.py BATTERS.csv PITCHERS.csv SALARY.csv
"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

TEAM_FIX = {"CWS": "CHW"}  # stats-file team codes -> Spotrac codes

BAT_COUNT_COLS = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS",
                  "BB", "SO", "SH", "SF", "HBP"]
PIT_COUNT_COLS = ["G", "GS", "CG", "SHO", "IP_decimal", "H", "ER", "K", "BB", "HR",
                  "W", "L", "SV", "BS", "HLD"]


# ------------------------------------------------------------------ helpers
def norm_name(name):
    """Make names comparable across sites: lowercase, no accents, no periods or
    apostrophes, no Jr./Sr./II/III.  'José Ramírez' -> 'jose ramirez',
    'T.J. Rumfield' -> 'tj rumfield', 'Luis Robert Jr.' -> 'luis robert'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[.']", "", s.lower())
    s = s.replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", s)
    return " ".join(s.split())


def ip_to_decimal(ip):
    """Baseball notation -> real innings. 200.2 means 200 2/3 innings."""
    whole = np.floor(ip)
    return whole + np.round((ip - whole) * 10) / 3


def decimal_to_ip(innings):
    """Real innings -> baseball notation. 200.667 -> 200.2"""
    whole = np.floor(innings + 1e-9)
    outs = np.round((innings - whole) * 3)
    return whole + outs / 10


def combine_traded(df, count_cols):
    """A player traded mid-season has one row per team. Add those rows together
    so each player has one row. Same name + same age = same player (this keeps
    two different players who share a name apart)."""
    df = df.sort_values("G", ascending=False)
    agg = {c: "sum" for c in count_cols}
    agg["Team"] = lambda teams: "/".join(teams)  # e.g. 'PIT/ATL', most games first
    if "Pos" in df.columns:
        agg["Pos"] = "first"  # position with the team he played most for
    return df.groupby(["Player", "Age"], as_index=False, sort=False).agg(agg)


def safe_div(num, den):
    return num / den.where(den > 0)


def round_half_up(x, digits):
    """Round like baseball sites do (6.075 -> 6.08). pandas' round() would give 6.07."""
    f = 10 ** digits
    return np.floor(x * f + 0.5 + 1e-9) / f


# ------------------------------------------------------------------ salary match
def attach_salary(stats, salary):
    """Left-join salary onto stats. Every stats row is kept; players with no
    Spotrac salary get blank salary columns.

    Matching order:
      1. Exact normalized name. If several Spotrac players share the name,
         pick the one on one of the player's 2026 teams.
      2. Nickname fallback (Louis/Louie, Nate/Nathan): same last name, same
         first initial, and on one of the player's teams.
    """
    salary = salary.copy()
    salary["key"] = salary["player"].map(norm_name)
    salary["last"] = salary["key"].str.split().str[-1]
    by_key = {k: g for k, g in salary.groupby("key")}
    # Names that belong to more than one player in the stats file (two different
    # 'Yunior Marte's) must also match on team, or both would get the same salary.
    stat_keys = stats["Player"].map(norm_name)
    shared_in_stats = set(stat_keys[stat_keys.duplicated(keep=False)])

    picked = []
    for _, row in stats.iterrows():
        key = norm_name(row["Player"])
        teams = set(row["Team"].split("/"))
        match = None

        cands = by_key.get(key)
        if cands is not None:
            if len(cands) == 1 and key not in shared_in_stats:
                match = cands.iloc[0]
            else:
                on_team = cands[cands["team"].isin(teams)]
                if len(on_team) == 1:
                    match = on_team.iloc[0]
        else:
            parts = key.split()
            cands = salary[(salary["last"] == parts[-1])
                           & (salary["key"].str[0] == parts[0][0])
                           & salary["team"].isin(teams)]
            if len(cands) == 1:
                match = cands.iloc[0]

        picked.append({
            "cap_total": match["cap_total"] if match is not None else np.nan,
            "salary_team": match["team"] if match is not None else None,
            "salary_position": match["position"] if match is not None else None,
            "salary_name": match["player"] if match is not None else None,
            "spotrac_id": match["spotrac_id"] if match is not None else np.nan,
        })
    return pd.concat([stats.reset_index(drop=True), pd.DataFrame(picked)], axis=1)


# ------------------------------------------------------------------ batters
def build_batters(bat, salary):
    bat = combine_traded(bat, BAT_COUNT_COLS)

    # Rate stats recomputed from the totals (needed for traded players)
    singles = bat["H"] - bat["2B"] - bat["3B"] - bat["HR"]
    total_bases = singles + 2 * bat["2B"] + 3 * bat["3B"] + 4 * bat["HR"]
    bat["PA"] = bat["AB"] + bat["BB"] + bat["HBP"] + bat["SF"] + bat["SH"]
    bat["AVG"] = round_half_up(safe_div(bat["H"], bat["AB"]), 3)
    bat["OBP"] = round_half_up(safe_div(bat["H"] + bat["BB"] + bat["HBP"],
                          bat["AB"] + bat["BB"] + bat["HBP"] + bat["SF"]), 3)
    bat["SLG"] = round_half_up(safe_div(total_bases, bat["AB"]), 3)
    bat["OPS"] = round_half_up((bat["OBP"] + bat["SLG"]), 3)

    bat = attach_salary(bat, salary)
    cols = (["Player", "Team", "Pos", "Age", "PA"] + BAT_COUNT_COLS
            + ["AVG", "OBP", "SLG", "OPS",
               "cap_total", "salary_team", "salary_position", "salary_name", "spotrac_id"])
    return bat[cols].sort_values("cap_total", ascending=False, na_position="last")


# ------------------------------------------------------------------ pitchers
def build_pitchers(pit, salary):
    pit = pit.copy()
    pit["IP_decimal"] = ip_to_decimal(pit["IP"])
    pit = combine_traded(pit, PIT_COUNT_COLS)

    ip = pit["IP_decimal"]
    pit["IP"] = decimal_to_ip(ip)              # baseball notation, like the original
    pit["IP_decimal"] = ip.round(3)            # real innings, use this for math
    pit["ERA"] = round_half_up((9 * safe_div(pit["ER"], ip)), 2)
    pit["WHIP"] = round_half_up(safe_div(pit["BB"] + pit["H"], ip), 2)

    # SP/RP from how he was actually used: SP if at least half his games were
    # starts. (Spotrac's labels are unreliable here, e.g. Roki Sasaki has 23
    # starts but is listed as RP.)
    pit["Role"] = np.where(pit["GS"] >= 0.5 * pit["G"], "SP", "RP")
    # Ohtani's G counts every game he played, including as a hitter, so the
    # rule above misreads him. All 14 of his pitching appearances were starts.
    pit.loc[pit["Player"] == "Shohei Ohtani", "Role"] = "SP"

    pit = attach_salary(pit, salary)
    cols = (["Player", "Team", "Role", "Age", "G", "GS", "CG", "SHO", "IP", "IP_decimal",
             "H", "ER", "K", "BB", "HR", "W", "L", "SV", "BS", "HLD", "ERA", "WHIP",
             "cap_total", "salary_team", "salary_position", "salary_name", "spotrac_id"])
    return pit[cols].sort_values("cap_total", ascending=False, na_position="last")


# ------------------------------------------------------------------ main
def main(batters_path, pitchers_path, salary_path):
    bat = pd.read_csv(batters_path, encoding="utf-8-sig")
    pit = pd.read_csv(pitchers_path, encoding="utf-8-sig")
    salary = pd.read_csv(salary_path, encoding="utf-8-sig")
    for df in (bat, pit):
        df["Team"] = df["Team"].replace(TEAM_FIX)

    batters = build_batters(bat, salary)
    pitchers = build_pitchers(pit, salary)

    batters.to_csv("batters_with_salary.csv", index=False)
    pitchers.to_csv("pitchers_with_salary.csv", index=False)

    for name, df in [("Batters", batters), ("Pitchers", pitchers)]:
        missing = df["cap_total"].isna().sum()
        print(f"{name}: {len(df)} players, {len(df) - missing} with salary, "
              f"{missing} without")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
