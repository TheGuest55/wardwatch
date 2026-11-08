import os
import time
import json
import random
import paho.mqtt.client as mqtt

BED_ID = os.environ.get("BED_ID", "bed1")
BROKER = "broker"
PORT = 1883
PASSWORD = os.environ.get("MQTT_PASS_WEARABLE", "change-me")

# Fix: Explicitly declare Callback API Version 1
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=f"wearable_{BED_ID}")
client.username_pw_set("wearable", PASSWORD)
client.connect(BROKER, PORT, 60)
client.loop_start()

while True:
    offline_time = random.randint(5, 10)
    time.sleep(offline_time)
    
    print(f"[{BED_ID}] Wearable connection restored, firing burst...", flush=True)
    for _ in range(5):
        payload = json.dumps({
            "heart_rate": random.randint(60, 100),
            "motion_index": round(random.uniform(0.1, 2.5), 2)
        })
        # Fix: Route to ward1/
        client.publish(f"ward1/{BED_ID}/wearable/vitals", payload)
        time.sleep(0.2)