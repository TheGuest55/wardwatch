"""The results bundle: exactly what the dashboard reads, the hard safety checks, and an end-to-end rebuild of the dashboard from
the bundle alone (real data: config.data_root(), credentials: data/raw/creds/accounts.json read in memory)."""
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from conftest import record, write
from helpers import run

from src.dashboard import bundle, config, loaders, metrics_view as mv

CREDS = Path(os.environ.get("WARDWATCH_CREDS_FILE") or config.REPO / "data" / "raw" / "creds" / "accounts.json")


def fake_creds(tmp_path):
    (tmp_path / "creddir").mkdir(exist_ok=True)
    p = tmp_path / "creddir" / "accounts.json"
    p.write_text(json.dumps({"alice-device": "Zq9-secretvalue-123456", "bob-device": "Yx8-othersecret-654321"}))
    return p


@pytest.fixture
def full(root):
    """The base corpus plus the extras a real data root has: predictions, incidents, and things that must NOT be bundled."""
    (root / "runs" / "lstm" / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"split": ["test"], "y_bin": [1], "score": [0.9], "pred": [1]}).to_parquet(
        root / "runs" / "lstm" / "predictions" / "A_k12_h32_s0.parquet")
    (root / "runs" / "lstm" / "curves").mkdir(exist_ok=True)
    (root / "runs" / "lstm" / "curves" / "A_k12_h32_s0.json").write_text(json.dumps({"history": [{"epoch": 1}]}))
    (root / "live" / "slices").mkdir(parents=True)
    (root / "live" / "slices" / "x.pcap").write_bytes(b"\xd4\xc3\xb2\xa1")
    (root / "live" / "incidents.jsonl").write_text(json.dumps({"incident_id": "r05-n-1", "session": "r05", "verdict": "No alert",
                                                                "pcap_slice": "live/slices/x.pcap", "pcap_sha256": "ab" * 32}) + "\n")
    (root / "runs" / "train.log").write_text("log line\n")
    (root / "models" / "expA").mkdir(parents=True)
    (root / "models" / "expA" / "m.json").write_text("{}")
    (root / "triage").mkdir()
    (root / "triage" / "peers_r05.parquet").write_bytes(b"x")
    pd.DataFrame({"split": ["train"], "label": ["benign"], "attack_class": [None]}).to_parquet(root / "testbed_windows.parquet")
    return root


class TestCollect:
    def test_includes_exactly_what_the_loaders_read(self, full):
        rels = bundle.collect(full)
        assert "runs/lstm/selected_config.json" in rels
        assert "runs/lstm/A_k12_h32_s0.json" in rels
        assert "runs/lstm/predictions/A_k12_h32_s0.parquet" in rels
        assert "runs/lstm/curves/A_k12_h32_s0.json" in rels
        assert "testbed_windows.parquet" in rels and "live/incidents.jsonl" in rels
        assert not any(r.endswith(("not_a_record.json", "broken.json")) for r in rels)           # not run records

    def test_excludes_slices_models_logs_peers_and_pcaps(self, full):
        rels = bundle.collect(full)
        for bad in ("live/slices/x.pcap", "runs/train.log", "models/expA/m.json", "triage/peers_r05.parquet"):
            assert bad not in rels
        bundle.check_paths(rels)                                                                  # does not raise


class TestHardChecks:
    def test_a_planted_pcap_or_secret_named_file_stops_the_build(self, full, tmp_path):
        for name in ("runs/lstm/leak.pcap", "runs/lstm/leak.pcapng"):
            with pytest.raises(SystemExit):
                bundle.check_paths(["runs/lstm/A_k12_h32_s0.json", name])
        for name in ("runs/creds.json", "runs/accounts.json", "runs/my_password.json", "runs/secret.json"):
            with pytest.raises(SystemExit):
                bundle.check_paths([name])
        # a pcap planted among the results stops the whole build, even though the collector would never select it
        for where in ("runs/lstm/predictions/planted.pcap", "runs/bootstrap/planted.pcapng", "live/planted.pcap"):
            p = full / where
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
            out = tmp_path / "out_pcap"
            with pytest.raises(SystemExit):
                bundle.build(full, out, fake_creds(tmp_path), commit="a" * 40)
            assert not out.exists() or not list(out.glob("*.zip"))
            p.unlink()
        (full / "runs" / "lstm" / "my_credentials.json").write_text("{}")
        with pytest.raises(SystemExit):
            bundle.build(full, tmp_path / "out_name", fake_creds(tmp_path), commit="a" * 40)
        (full / "runs" / "lstm" / "my_credentials.json").unlink()

    def test_a_planted_credential_string_stops_the_build_and_writes_nothing(self, full, tmp_path):
        for where, name in (("text", "runs/lstm/A_k12_h32_s0.json"), ("user", "runs/lstm/A_k1_h32_s0.json")):
            p = full / name
            obj = json.loads(p.read_text())
            obj["tags"]["note"] = "found Zq9-secretvalue-123456 here" if where == "text" else "device alice-device sent"
            p.write_text(json.dumps(obj))
            out = tmp_path / f"out_{where}"
            with pytest.raises(SystemExit) as e:
                bundle.build(full, out, fake_creds(tmp_path), commit="b" * 40)
            assert "Zq9-secretvalue" not in str(e.value) and "alice-device" not in str(e.value)    # never printed
            assert not out.exists() or not list(out.glob("*.zip"))
            p.write_text(json.dumps({**obj, "tags": {k: v for k, v in obj["tags"].items() if k != "note"}}))

    def test_credentials_file_without_entries_cannot_run_the_scan(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text("{}")
        with pytest.raises(SystemExit):
            bundle.load_secrets(p)

    def test_the_credentials_file_is_never_copied(self, full, tmp_path):
        out = tmp_path / "out"
        r = bundle.build(full, out, fake_creds(tmp_path), commit="c" * 40)
        names = zipfile.ZipFile(r["zip"]).namelist()
        assert not any("accounts" in n or "creds" in n for n in names)


class TestOutputsAndManifest:
    def test_zip_manifest_and_bundle_json(self, full, tmp_path):
        out = tmp_path / "out"
        r = bundle.build(full, out, fake_creds(tmp_path), commit="d" * 40)
        assert Path(r["zip"]).name == "wardwatch_results_ddddddd.zip"
        info = json.loads((out / "BUNDLE.json").read_text())
        assert info["commit"] == "d" * 40 and info["file_count"] == r["files"] and info["created_utc"].endswith("Z")
        assert info["zip_sha256"] == r["sha256"]
        unpack = tmp_path / "unpacked"
        zipfile.ZipFile(r["zip"]).extractall(unpack)
        assert (unpack / "BUNDLE.json").exists() and (unpack / "MANIFEST.sha256").exists()
        assert bundle.verify_manifest(unpack) == []
        assert (out / "MANIFEST.sha256").read_text() == (unpack / "MANIFEST.sha256").read_text()
        assert len(list(p for p in unpack.rglob("*") if p.is_file())) == r["files"]

    def test_manifest_detects_a_changed_a_missing_and_an_extra_file(self, full, tmp_path):
        r = bundle.build(full, tmp_path / "out", fake_creds(tmp_path), commit="e" * 40)
        unpack = tmp_path / "u"
        zipfile.ZipFile(r["zip"]).extractall(unpack)
        (unpack / "runs" / "lstm" / "selected_config.json").write_text("{}")
        (unpack / "testbed_windows.parquet").unlink()
        (unpack / "extra.txt").write_text("x")
        probs = " ".join(bundle.verify_manifest(unpack))
        assert "hash differs: runs/lstm/selected_config.json" in probs and "missing: testbed_windows.parquet" in probs
        assert "not in the manifest: extra.txt" in probs

    def test_same_inputs_give_the_same_files_and_manifest(self, full, tmp_path):
        a = bundle.build(full, tmp_path / "a", fake_creds(tmp_path), commit="f" * 40)
        b = bundle.build(full, tmp_path / "b", fake_creds(tmp_path), commit="f" * 40)
        assert (tmp_path / "a" / "MANIFEST.sha256").read_text() == (tmp_path / "b" / "MANIFEST.sha256").read_text()
        za, zb = zipfile.ZipFile(a["zip"]), zipfile.ZipFile(b["zip"])
        assert za.namelist() == zb.namelist()


class TestIncidentsPageInABundle:
    def _incidents(self, root, monkeypatch):
        at = run("src/dashboard/pages/6_Incidents.py", root, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        return " ".join(i.value for i in at.info)

    def test_missing_slice_in_a_bundle_says_not_included_and_keeps_the_hash(self, full, tmp_path, monkeypatch):
        r = bundle.build(full, tmp_path / "out", fake_creds(tmp_path), commit="1" * 40)
        unpack = tmp_path / "bundle_root"
        zipfile.ZipFile(r["zip"]).extractall(unpack)
        monkeypatch.setenv("WARDWATCH_DATA_ROOT", str(unpack))
        at = run("src/dashboard/pages/6_Incidents.py", unpack, monkeypatch)
        assert not at.exception, [e.value for e in at.exception]
        info = " ".join(i.value for i in at.info)
        assert "evidence file not included in this bundle (PCAP slices hold lab credentials)" in info
        assert "ab" * 32 in " ".join(c.value for c in at.code)                          # the recorded sha256 is still shown

    def test_without_bundle_json_the_old_message_stays(self, full, monkeypatch):
        info = self._incidents(full, monkeypatch)                                        # the full root has the slice dir but no file x? it exists:
        assert "not included in this bundle" not in info


# ------------------------------------------------------------------------------- real data, end to end
REAL_MISSING = ([] if (config.data_root() / "runs" / "expA").is_dir() else ["the recorded runs/ (results zip)"]) + \
    ([] if CREDS.is_file() else ["the lab credentials file (not published)"])


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory):
    src = config.data_root()
    t = tmp_path_factory.mktemp("real")
    r = bundle.build(src, t / "out", CREDS, commit="0" * 40)
    empty = t / "empty_root"
    empty.mkdir()
    zipfile.ZipFile(r["zip"]).extractall(empty)
    return src, empty, r


@pytest.mark.skipif(bool(REAL_MISSING), reason="needs " + " and ".join(REAL_MISSING) + ": see USER_MANUAL Level 3 (building the bundle needs the credentials file)")
class TestRealData:
    def test_manifest_verifies_and_nothing_forbidden_is_inside(self, real_bundle):
        src, empty, r = real_bundle
        assert bundle.verify_manifest(empty) == []
        names = zipfile.ZipFile(r["zip"]).namelist()
        assert not any(n.lower().endswith((".pcap", ".pcapng", ".log")) for n in names)
        assert not any(("slices" in n.split("/")) or ("models" in n.split("/")) or n.startswith("triage/") for n in names)
        assert not any("accounts" in n.lower() or "creds" in n.lower() for n in names)
        assert r["files"] == len(names)

    def test_headline_and_selection_equal_the_full_data_root(self, real_bundle):
        src, empty, _ = real_bundle
        a, b = loaders.load_records(src).data, loaders.load_records(empty).data
        assert len(a) == len(b)
        ha, hb = mv.headline(a, "all", src), mv.headline(b, "all", empty)
        pd.testing.assert_frame_equal(ha, hb)
        for view in ("all", "noatk"):
            pd.testing.assert_frame_equal(mv.headline(a, view, src), mv.headline(b, view, empty))
        assert mv.selection(ha) == mv.selection(hb)
        pd.testing.assert_frame_equal(mv.c_sensitivity(a, "all"), mv.c_sensitivity(b, "all"))

    def test_all_pages_load_from_an_empty_root_holding_only_the_bundle(self, real_bundle, monkeypatch):
        _, empty, _ = real_bundle
        pages = [Path("src/dashboard/app.py"), *sorted(Path("src/dashboard/pages").glob("*.py"))]
        assert len(pages) == 7
        for page in pages:
            at = run(str(page), empty, monkeypatch)
            assert not at.exception, (page.name, [e.value for e in at.exception])

    def test_incidents_page_shows_the_not_included_message(self, real_bundle, monkeypatch):
        _, empty, _ = real_bundle
        at = run("src/dashboard/pages/6_Incidents.py", empty, monkeypatch)
        assert not at.exception
        assert any("evidence file not included in this bundle" in i.value for i in at.info)
