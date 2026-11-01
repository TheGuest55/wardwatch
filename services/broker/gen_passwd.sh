#!/bin/sh
# Builds the password file from the environment, then starts the broker.
set -eu
f=/tmp/passwd
: > "$f"
chmod 0700 "$f"
mosquitto_passwd -b "$f" nurse-station "$MQTT_PASS_NURSE_STATION"
mosquitto_passwd -b "$f" patient-monitor "$MQTT_PASS_PATIENT_MONITOR"
mosquitto_passwd -b "$f" infusion-pump "$MQTT_PASS_INFUSION_PUMP"
mosquitto_passwd -b "$f" wearable "$MQTT_PASS_WEARABLE"
chown mosquitto:mosquitto "$f"
exec mosquitto -c /mosquitto/config/mosquitto.conf
