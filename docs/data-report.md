> **Verdict:** REVIEW — a run has no window or an orphan run_id exists

# WardWatch testbed dataset — data report

- rows: **83994**  |  features: **52**  |  sessions: n01, s01, s02, s03, s04, s05, s06

## Rows per class x session

Label counts per session (attack windows broken out by class below).

```
session         n01    s01    s02    s03    s04    s05    s06
cls                                                          
arp_mitm          0     60     57     60     57     57     60
benign        16872  10597  10937  10312  10984  10961  11566
bruteforce        0      8      6      2      0      8     54
eavesdrop         0     70     70     70     70     70     70
mqtt_flood        0     10     10     10     10     10     10
pump_cmd          0     18     18     18     20     18     18
recon_scan        0     80     96     96     96     96     80
spoof_vitals      0     22     22     22     22     22     22
syn_flood         0     12     12     12     12     10     12
```

- attack rows: 1765 (2.1%); benign rows: 82229 (97.9%)

## Clean attack windows per class x session

Attack windows only (benign excluded; boundary windows already dropped). A 0 means that class produced no clean labelled window in that session.

```
session       s01  s02  s03  s04  s05  s06  TOTAL
attack_class                                     
arp_mitm       60   57   60   57   57   60    351
bruteforce      8    6    2    0    8   54     78
eavesdrop      70   70   70   70   70   70    420
mqtt_flood     10   10   10   10   10   10     60
pump_cmd       18   18   18   20   18   18    110
recon_scan     80   96   96   96   96   80    544
spoof_vitals   22   22   22   22   22   22    132
syn_flood      12   12   12   12   10   12     70
```

- (class, session) with zero clean windows: [('bruteforce', 's04')]

## Every class present in train / val / test

- `arp_mitm`: {'train': 177, 'val': 117, 'test': 57}  -> OK
- `bruteforce`: {'train': 16, 'val': 54, 'test': 8}  -> OK
- `eavesdrop`: {'train': 210, 'val': 140, 'test': 70}  -> OK
- `mqtt_flood`: {'train': 30, 'val': 20, 'test': 10}  -> OK
- `pump_cmd`: {'train': 54, 'val': 38, 'test': 18}  -> OK
- `recon_scan`: {'train': 272, 'val': 176, 'test': 96}  -> OK
- `spoof_vitals`: {'train': 66, 'val': 44, 'test': 22}  -> OK
- `syn_flood`: {'train': 36, 'val': 24, 'test': 10}  -> OK

## Ground-truth run coverage

Every logged run should produce >=1 attack window; no attack window should carry a run_id that isn't in ground truth.

- s01: 8 runs, 8 with >=1 window
- s02: 8 runs, 8 with >=1 window
- s03: 8 runs, 8 with >=1 window
- s04: 8 runs, 7 with >=1 window; NO WINDOW: ['s04-bruteforce-04']
- s05: 8 runs, 8 with >=1 window
- s06: 8 runs, 8 with >=1 window

## Benign events per session

Reboots + nurse reconnects keep benign wildcard-SUBSCRIBE recurring; legit nurse commands keep pump-command traffic in benign data. A session with zero reconnects or zero commands is flagged.

- n01: {'reboot': 52, 'reconnect': 39, 'nurse_command': 22}
- s01: {'reboot': 30, 'reconnect': 25, 'nurse_command': 15}
- s02: {'reboot': 29, 'reconnect': 26, 'nurse_command': 15}
- s03: {'reboot': 28, 'reconnect': 24, 'nurse_command': 14}
- s04: {'reboot': 30, 'reconnect': 26, 'nurse_command': 15}
- s05: {'reboot': 29, 'reconnect': 26, 'nurse_command': 15}
- s06: {'reboot': 34, 'reconnect': 27, 'nurse_command': 17}

## Data hygiene

- hosts per window: min 4, median 12, max 16
- infinite feature cells: 0
- columns with NaN: none

## Boundary windows dropped (per session)

- {'n01': 0, 's01': 216, 's02': 204, 's03': 201, 's04': 202, 's05': 213, 's06': 209}

## Leakage probe

Binary attack-vs-normal F1. `metadata_only` uses ONLY identifiers + time (host_ip, host_role, window_start) — these are excluded from the model; a high score here is the leakage we avoid by keeping them as metadata. `features_only` uses the real model inputs.

```
{
  "metadata_only": {
    "f1": 0.4343,
    "accuracy": 0.9812
  },
  "split_fallback_random": false,
  "features_only": {
    "f1": 0.9495,
    "accuracy": 0.9975,
    "top15": [
      [
        "pkts_total",
        0.114
      ],
      [
        "nf_mean_flow_dur",
        0.1119
      ],
      [
        "pkts_recv",
        0.1079
      ],
      [
        "bytes_recv",
        0.0807
      ],
      [
        "bytes_sent",
        0.0729
      ],
      [
        "std_iat",
        0.0686
      ],
      [
        "bytes_total",
        0.0602
      ],
      [
        "pkts_sent",
        0.0521
      ],
      [
        "mean_iat",
        0.048
      ],
      [
        "mean_pkt_size",
        0.0249
      ],
      [
        "frac_tcp",
        0.0245
      ],
      [
        "min_pkt_size",
        0.0207
      ],
      [
        "ip_mac_conflict",
        0.0196
      ],
      [
        "topic_is_command",
        0.0187
      ],
      [
        "nf_active_flows",
        0.0186
      ]
    ]
  },
  "per_class_recall": {
    "recall": {
      "arp_mitm": 0.9474,
      "bruteforce": 1.0,
      "eavesdrop": 0.7143,
      "mqtt_flood": 1.0,
      "pump_cmd": 0.8333,
      "recon_scan": 1.0,
      "spoof_vitals": 0.9091,
      "syn_flood": 1.0
    },
    "benign_specificity": 1.0,
    "test_support": {
      "arp_mitm": 57,
      "bruteforce": 8,
      "eavesdrop": 70,
      "mqtt_flood": 10,
      "pump_cmd": 18,
      "recon_scan": 96,
      "spoof_vitals": 22,
      "syn_flood": 10
    }
  },
  "features_only_no_attacker": {
    "f1": 0.9714,
    "accuracy": 0.9989
  }
}
```

### Features-only per-class recall (test split)

Binary classifier recall on each attack class's test windows (benign shown as specificity). A low value is a hard class the single probe misses even at overall F1 0.9495 — expected for stolen-credential classes.

- `arp_mitm`: recall 0.9474 (n=57)
- `bruteforce`: recall 1.0 (n=8)
- `eavesdrop`: recall 0.7143 (n=70)
- `mqtt_flood`: recall 1.0 (n=10)
- `pump_cmd`: recall 0.8333 (n=18)
- `recon_scan`: recall 1.0 (n=96)
- `spoof_vitals`: recall 0.9091 (n=22)
- `syn_flood`: recall 1.0 (n=10)
- benign specificity: 1.0


- split used: time-based train/test
- metadata-only F1: {'f1': 0.4343, 'accuracy': 0.9812}
- any metadata column among top-15 features: no

