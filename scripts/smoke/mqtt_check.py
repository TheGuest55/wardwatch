"""Runs inside the patient-monitor container: confirms the broker rejects a wrong
password and enforces the ACL (the monitor may not publish to the pump command topic,
but may publish to its own). This verifies the lockdown, it does not bypass it."""
import os
import sys
import threading

import paho.mqtt.client as mqtt

HOST = "172.28.10.10"
USER = os.environ["MQTT_USERNAME"]
PASSWORD = os.environ["MQTT_PASSWORD"]


def new_client(password):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    client.username_pw_set(USER, password)
    return client


def connect_reason(password):
    client = new_client(password)
    result, done = {}, threading.Event()

    def on_connect(c, userdata, flags, reason_code, properties):
        result["reason"] = reason_code
        done.set()

    client.on_connect = on_connect
    client.connect(HOST, 1883)
    client.loop_start()
    done.wait(10)
    client.loop_stop()
    return result.get("reason")


def publish_reason(topic):
    client = new_client(PASSWORD)
    result, connected, acked = {}, threading.Event(), threading.Event()

    def on_publish(c, userdata, mid, reason_code, properties):
        result["reason"] = reason_code
        acked.set()

    client.on_connect = lambda *a: connected.set()
    client.on_publish = on_publish
    client.connect(HOST, 1883)
    client.loop_start()
    connected.wait(10)
    client.publish(topic, "smoke-test", qos=1)
    acked.wait(10)
    client.loop_stop()
    return result.get("reason")


wrong = connect_reason(PASSWORD + "-wrong")
right = connect_reason(PASSWORD)
denied = publish_reason("ward1/bed1/pump/cmd")
allowed = publish_reason("ward1/bed1/monitor/smoke")
print(f"wrong_password={wrong}; right_password={right}; pump_cmd={denied}; own_topic={allowed}")

ok = (
    wrong is not None and wrong.is_failure
    and right is not None and not right.is_failure
    and denied is not None and denied.is_failure
    and allowed is not None and not allowed.is_failure
)
sys.exit(0 if ok else 1)
