# Session-isolation guards (review M1). Sourced by run_session.sh. Functions use $DOCKER so tests
# can stub the docker CLI, and operate read-only (they never touch data/raw).
#
#   project_containers_present   -> true if any wardwatch compose container exists (up OR exited).
#   pcap_changed_within DIR SECS  -> true if any file under DIR was modified within SECS seconds
#                                   (catches a second, external capture writing data/raw/pcap).
DOCKER="${DOCKER:-docker}"
PROJECT_LABEL="com.docker.compose.project=wardwatch"

project_containers() {
  "$DOCKER" ps -aq --filter "label=$PROJECT_LABEL" 2>/dev/null
}

project_containers_present() {
  [ -n "$(project_containers)" ]
}

pcap_changed_within() {
  dir="$1"; secs="$2"
  [ -d "$dir" ] || return 1
  mins=$(awk "BEGIN{printf \"%.4f\", $secs/60}")
  [ -n "$(find "$dir" -type f -mmin "-$mins" 2>/dev/null | head -n1)" ]
}
