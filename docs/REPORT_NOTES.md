# REPORT_NOTES

Report limit: 8 pages excl. references and appendix.
Section tags: [DATA] dataset & testbed · [PREP] pre-processing · [MODEL] models · [EVAL] evaluation & results ·
[FORENSICS] triage & incidents · [RT] real-time & dashboard · [LIMIT] limitations · [APPENDIX] detail for appendix ·
[ENV] environment (usually appendix / user manual)

## Page budget (target)
| Section | Pages |
|---|---|
| Title, abstract, introduction | 1 |
| Related work | 1 |
| Dataset & testbed, system diagram | 1.5 |
| Pre-processing & features | 0.5 |
| Models & evaluation design | 1 |
| Results | 2 |
| Discussion, limitations, conclusion | 1 |

## Log

### 2026-11-01 · [ENV] · success
**What:** Task 1 complete: repo skeleton, `.gitignore`, `.env.example`, README stub and this file created.
**Why:** docs/environment.md B4 Task 1.
**Evidence:** `git log` commit "env: add repo skeleton"; `.gitkeep` files hold the empty tracked directories.
**Report impact:** Appendix / user manual only.

### 2026-11-01 · [ENV] · failure
**What:** Tasks 2-6 blocked: all required hosts return HTTP 403 "Approval required", including `registry-1.docker.io`, `auth.docker.io`, `production.cloudflare.docker.com`, `pypi.org`, `files.pythonhosted.org`, `physionet.org`, `github.com` and `deb.debian.org`. `docker pull hello-world` also fails with 403. Host allowlist from environment.md A3 is not in effect for this sandbox.
**Why:** Network policy is set on the host and cannot be changed from the sandbox; B2 says stop and ask.
**Evidence:** `curl -s -m 15 -w '\n[%{http_code}]\n' <url>`; `docker pull hello-world`. No images are present (`docker images` is empty).
**Report impact:** None unless it persists; appendix at most.

### 2026-11-01 · [ENV] · decision
**What:** Sandbox baseline recorded: Docker 29.9.2, Compose v5.6.1, 6 CPUs, about 11 GB RAM (`free -g`), 9.8 GB Docker storage (`df -h /var/lib/docker`), 364 GB free on the project mount. The VM has Python 3.14.4 and no `python3.11`, `tcpdump` or `tshark`.
**Why:** B1 rule 6 requires Python 3.11, so Python code will run in pinned 3.11 containers instead of the VM's interpreter, and pcap counting will also run in a container.
**Evidence:** `docker version`, `docker compose version`, `df -h`, `nproc`, `free -g`, `which python3.11 tcpdump tshark`.
**Report impact:** Appendix / user manual only.
