"""Read-only diagnosis of the incident grouping. It changes no rule and writes no incident.

    python -m src.triage.diagnose_grouping [--for-report]

Why was grouping accuracy 0.25 (purity 1.0)? This rebuilds the incidents of the held-out sessions with the SAME code
and the (key, gap) recorded in runs/triage/triage_A.json (no post-alert step), asserts they equal
triage/incidents_offline.jsonl (id, first_seen, last_seen, source, target_host, n_alerts, verdict, cell: any difference
is a stop), then explains every ground-truth attack run:

  * n_incidents, and the MAIN incident (the one holding most of the run's alerted attack windows);
  * one cause per extra incident, in this order:
      attribution  the incident's source MAC differs from the main incident's (the source-role transition is
                   recorded, e.g. attacker -> broker);
      gap          same source MAC, separated from the main incident by more than the gap;
      other        anything else: flagged, not explained;
  * whether the run's incidents also hold other runs' windows, benign alerts, or victim/boundary alerts.
False-alarm incidents (no attack-labelled alerted window) are split into benign-only and "with victim/boundary" and
counted by source role and by verdict / cell.

Roles here come from the stored parquet's ground-truth host_role: EVALUATION ONLY, to describe what happened. They
are never an input to any attribution or grouping decision.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .. import common
from ..models import tracking
from ..models import train_tree as tt
from . import incidents as inc_mod
from . import offline

COMPARE = ("incident_id", "first_seen", "last_seen", "source", "target_host", "n_alerts", "verdict", "cell")
CAUSES = ("attribution", "gap", "other")
RUNS = offline.RUNS


def assert_equal_to_offline(records, offline_records):
    """The rebuilt incidents must equal the stored offline ones on the fields above."""
    got = {r["incident_id"]: r for r in records}
    want = {r["incident_id"]: r for r in offline_records}
    probs = []
    if len(got) != len(records):
        probs.append("duplicate incident ids in the rebuild")
    if set(got) != set(want):
        probs.append(f"ids differ: only rebuilt {sorted(set(got) - set(want))[:5]}, "
                     f"only stored {sorted(set(want) - set(got))[:5]}")
    for k in sorted(set(got) & set(want)):
        for f in COMPARE:
            if got[k].get(f) != want[k].get(f):
                probs.append(f"{k}.{f}: rebuilt {got[k].get(f)!r} != stored {want[k].get(f)!r}")
    if probs:
        raise AssertionError("; ".join(probs[:8]))
    return len(got)


class _Alerts:
    """Per-alert facts for one session: label, run, attributed source/target MAC (from the same attribute() the
    Triager used), and the evaluation-only role of a MAC."""

    def __init__(self, sess):
        df = sess.df
        self.label = {(m, int(w)): (lab, run) for m, w, lab, run in
                      zip(df["host_mac"], df["window_start_epoch"], df["label"], df["run_id"])}
        self.role = (df.groupby("host_mac")["host_role"].agg(lambda s: s.mode().iat[0]).to_dict())
        self.peers = sess.peers

    def source_target(self, mac, win):
        return inc_mod.attribute(mac, self.peers.get((mac, win), {}))


def _incident_facts(tr, facts):
    """iid -> dict(source_mac, first, last, alerts=[(mac, win, label, run)])."""
    recs = {r["incident_id"]: r for r in tr.records}
    out = {}
    for iid, mem in tr.members.items():
        al = [SimpleNamespace(host_mac=m, window_start=w, source=facts.source_target(m, w)[0], target=facts.source_target(m, w)[1])
              for m, w in mem]
        src = Counter(a.source for a in al).most_common()
        source = sorted(m for m, c in src if c == src[0][1])[0]
        wins = [w for _, w in mem]
        out[iid] = {"source": source, "first": min(wins), "last": max(wins), "rec": recs[iid],
                    "alerts": [(m, w) + facts.label.get((m, w), (None, None)) for m, w in mem]}
    return out


def diagnose(sess, tr, key, gap):
    facts = _Alerts(sess)
    inc = _incident_facts(tr, facts)
    W = inc_mod.W
    runs_meta = {iv["run_id"]: iv["class"] for iv in sess.intervals}
    per_run = {}
    for run, cls in sorted(runs_meta.items()):
        mine = {iid: sum(1 for a in v["alerts"] if a[2] == "attack" and a[3] == run) for iid, v in inc.items()}
        mine = {i: n for i, n in mine.items() if n}
        row = {"class": cls, "n_alerted_windows": int(sum(mine.values())), "n_incidents": len(mine), "extra": []}
        if not mine:
            per_run[run] = {**row, "main_incident": None, "detected": False}
            continue
        main = sorted(mine, key=lambda i: (-mine[i], inc[i]["first"], i))[0]
        row.update(detected=True, main_incident=main, main_alerts=mine[main],
                   main_source_role=facts.role.get(inc[main]["source"], "unknown"))
        for iid in sorted(mine, key=lambda i: inc[i]["first"]):
            if iid == main:
                continue
            e, m = inc[iid], inc[main]
            if e["source"] != m["source"]:
                cause = "attribution"
            else:
                sep = (e["first"] - (m["last"] + W)) if e["first"] >= m["first"] else (m["first"] - (e["last"] + W))
                cause = "gap" if sep > gap else "other"
            src_role = facts.role.get(e["source"], "unknown")
            row["extra"].append({"incident_id": iid, "cause": cause, "alerts_for_run": mine[iid],
                                 "source_role": src_role,
                                 "transition": f"{row['main_source_role']}->{src_role}" if cause == "attribution" else None})
        others = lambda v: {a[3] for a in v["alerts"] if a[2] == "attack" and a[3] != run}
        row["mixes_other_runs"] = sorted({r for i in mine for r in others(inc[i])})
        row["incidents_with_benign_alerts"] = sum(1 for i in mine if any(a[2] == "benign" for a in inc[i]["alerts"]))
        row["incidents_with_victim_boundary"] = sum(1 for i in mine if any(a[2] in ("victim", "boundary") for a in inc[i]["alerts"]))
        per_run[run] = row
    false_alarm = {"n": 0, "benign_only": 0, "with_victim_boundary": 0, "by_source_role": Counter(),
                   "by_verdict": Counter(), "by_cell": Counter(), "by_source_role_and_verdict": Counter()}
    for iid, v in inc.items():
        if any(a[2] == "attack" for a in v["alerts"]):
            continue
        false_alarm["n"] += 1
        vb = any(a[2] in ("victim", "boundary") for a in v["alerts"])
        false_alarm["with_victim_boundary" if vb else "benign_only"] += 1
        role = facts.role.get(v["source"], "unknown")
        false_alarm["by_source_role"][role] += 1
        false_alarm["by_verdict"][v["rec"]["verdict"]] += 1
        false_alarm["by_cell"][v["rec"]["cell"]] += 1
        false_alarm["by_source_role_and_verdict"][f"{role} | {v['rec']['verdict']}"] += 1
    false_alarm = {k: (dict(sorted(x.items())) if isinstance(x, Counter) else x) for k, x in false_alarm.items()}
    causes = Counter(e["cause"] for r in per_run.values() for e in r["extra"])
    trans = Counter(e["transition"] for r in per_run.values() for e in r["extra"] if e["transition"])
    return {"runs": per_run, "false_alarm": false_alarm, "n_incidents": len(inc),
            "extra_by_cause": {c: causes.get(c, 0) for c in CAUSES},
            "extra_by_transition": dict(sorted(trans.items())),
            "runs_detected": sum(1 for r in per_run.values() if r["detected"]),
            "runs_single_incident": sum(1 for r in per_run.values() if r["detected"] and r["n_incidents"] == 1)}


def run(for_report=False, out_dir=RUNS, tri_dir=offline.TRI, a_record=RUNS / "triage_A.json", sessions=offline.TEST):
    from ..realtime.pipeline import Pair
    a = json.loads(Path(a_record).read_text())
    key, gap = a["params"]["grouping_key"], a["params"]["grouping_gap_s"]
    roles = common.ip_role_map(common.load_config())
    stored = [json.loads(l) for l in (Path(tri_dir) / "incidents_offline.jsonl").read_text().splitlines() if l.strip()]
    pair = Pair()
    per, equal = {}, {}
    for n in sessions:
        s = offline.Session(n, pair, peers_dir=tri_dir)
        tr = offline.run_triager(s, key, gap, roles)                    # no post step: no SHAP, no slices
        try:
            equal[n] = assert_equal_to_offline(tr.records, [r for r in stored if r["session"] == n])
        except AssertionError as e:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "STOP.md").write_text(f"# STOP: grouping diagnosis, session {n}\n\n{e}\n")
            raise
        per[n] = diagnose(s, tr, key, gap)
        print(f"[diagnose] {n}: {equal[n]} incidents equal the offline ones; "
              f"{per[n]['runs_single_incident']}/{per[n]['runs_detected']} runs in a single incident; "
              f"extra incidents by cause {per[n]['extra_by_cause']}", flush=True)
    tot_c = Counter()
    tot_t = Counter()
    for r in per.values():
        tot_c.update(r["extra_by_cause"])
        tot_t.update(r["extra_by_transition"])
    metrics = {f"extra_{c}": tot_c.get(c, 0) for c in CAUSES}
    metrics.update({"n_incidents": sum(r["n_incidents"] for r in per.values()),
                    "false_incidents": sum(r["false_alarm"]["n"] for r in per.values()),
                    "runs_detected": sum(r["runs_detected"] for r in per.values()),
                    "runs_single_incident": sum(r["runs_single_incident"] for r in per.values())})
    # seed 0: the deployed pair is the Experiment A seed-0 models (a report-grade record needs a non-null seed)
    rec = tracking.build_run_record(
        "triage", "xgboost+isolation_forest", "pair",
        params={"grouping_key": key, "grouping_gap_s": gap, "read_only": True, "rule_changed": False},
        metrics=metrics,
        tags={"stage": "grouping_diagnosis", "sessions": list(sessions), "equal_to_offline": equal,
              "per_session": per, "totals": {"by_cause": dict(tot_c), "by_transition": dict(sorted(tot_t.items()))},
              "roles_note": "roles are ground-truth host_role, evaluation only; no decision reads them"},
        seed=0, split_path=str(tt.SPLIT), dataset_path=str(tt.DATASET), for_report=for_report)
    out_dir = Path(out_dir)
    tracking.log_run(rec, json_path=str(out_dir / "grouping_diagnosis.json"), use_mlflow=False)
    (out_dir / "grouping_diagnosis.md").write_text(summary_md(rec))
    print(f"[diagnose] causes {dict(tot_c)}; transitions {dict(tot_t)}; record {out_dir / 'grouping_diagnosis.json'}", flush=True)
    return rec


def summary_md(rec):
    t = rec["tags"]
    L = ["# Grouping diagnosis (read-only; the grouping rule is unchanged)", "",
         f"key `{rec['params']['grouping_key']}`, gap {rec['params']['grouping_gap_s']} s. Rebuilt incidents equal the offline ones: "
         f"{t['equal_to_offline']}. Roles are ground-truth host_role, evaluation only.", ""]
    for n, r in t["per_session"].items():
        L += [f"## {n}", "", f"{r['runs_single_incident']}/{r['runs_detected']} detected runs sit in one incident; "
              f"{r['n_incidents']} incidents; extra incidents by cause {r['extra_by_cause']}; by transition {r['extra_by_transition']}.", "",
              "| run | class | alerted windows | incidents | main alerts | extra by cause | mixes other runs | benign-alert incidents | victim/boundary incidents |",
              "|---|---|---|---|---|---|---|---|---|"]
        for run, x in r["runs"].items():
            c = Counter(e["cause"] for e in x["extra"])
            L.append(f"| {run} | {x['class']} | {x['n_alerted_windows']} | {x['n_incidents']} | {x.get('main_alerts', '-')} | "
                     f"{dict(c) or '-'} | {x.get('mixes_other_runs') or '-'} | {x.get('incidents_with_benign_alerts', '-')} | "
                     f"{x.get('incidents_with_victim_boundary', '-')} |")
        fa = r["false_alarm"]
        L += ["", f"False-alarm incidents (no attack window): {fa['n']} ({fa['benign_only']} benign-only, "
              f"{fa['with_victim_boundary']} with victim/boundary windows). By source role {fa['by_source_role']}; "
              f"by verdict {fa['by_verdict']}; by cell {fa['by_cell']}.", ""]
    L += [f"Totals: extra incidents by cause {t['totals']['by_cause']}; by role transition {t['totals']['by_transition']}.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--for-report", action="store_true")
    run(for_report=ap.parse_args(argv).for_report)


if __name__ == "__main__":
    main()
