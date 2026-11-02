"""Credential resolution for the abuse classes, in its own module to avoid a circular import
with the attacks package __init__."""


def pick_creds(ctx, role):
    """Prefer creds recovered earlier this session (stolen_creds.json); otherwise fall back to
    the provisioned real password, flagged so ground truth records creds_source=fallback.
    role is 'nurse' or 'device'. Returns (username, password, source)."""
    stolen = ctx.load_stolen()
    want_nurse = role == "nurse"
    for user, rec in stolen.items():
        if (user == ctx.nurse_user) == want_nurse:
            return user, rec["password"], "stolen"
    user = ctx.nurse_user if want_nurse else ctx.rng.choice(ctx.monitor_users)
    pw = ctx.provisioned(user)
    if pw:
        return user, pw, "fallback"
    return None, None, "unavailable"
