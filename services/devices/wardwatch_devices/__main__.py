import os
import sys

from . import monitor, nurse, pump, wearable

ROLES = {
    "patient-monitor": monitor.run,
    "infusion-pump": pump.run,
    "wearable": wearable.run,
    "nurse-station": nurse.run,
}


def main():
    role = os.environ.get("DEVICE_ROLE", "")
    if role not in ROLES:
        sys.exit(f"DEVICE_ROLE must be one of {sorted(ROLES)}, got {role!r}")
    ROLES[role]()


main()
