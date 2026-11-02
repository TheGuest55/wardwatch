"""recon_scan: a bounded nmap SYN + service scan of the ward subnet."""
from .. import util


def recon_scan(ctx, params):
    rate = int(params.get("max_rate", 500))
    timeout = int(params.get("timeout", 120))
    subnet = ctx.cfg["network"]["subnet"]
    rc, out, _ = util.run(
        ["nmap", "-sS", "-sV", "-T3", "--max-rate", str(rate), "-oG", "-", subnet],
        timeout=timeout)
    hosts_up = out.count("Status: Up") if out else 0
    return {"targets": [subnet],
            "notes": f"nmap -sS -sV max-rate={rate} rc={rc} hosts_up~{hosts_up}"}


recon_scan.STAGE = "reconnaissance"
