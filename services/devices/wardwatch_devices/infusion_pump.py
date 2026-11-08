import os
import time
import json
import paho.mqtt.client as mqtt

BED_ID = os.environ.get("BED_ID", "bed1")
BROKER = "broker"
PORT = 1883
PASSWORD = os.environ.get("MQTT_PASS_INFUSION_PUMP", "change-me")

state = {"rate_ml_hr": 50, "volume_remaining_ml": 1000}

def on_connect(client, userdata, flags, rc):
    print(f"Pump connected with result code {rc}", flush=True)
    client.subscribe(f"ward1/{BED_ID}/pump/command")

def on_message(client, userdata, msg):
    try:
        command = json.loads(msg.payload.decode())
        if "new_rate" in command:
            state["rate_ml_hr"] = command["new_rate"]
            print(f"[{BED_ID}] Rate updated to {state['rate_ml_hr']} ml/hr", flush=True)
    except Exception as e:
        print(f"Invalid command received: {e}", flush=True)

# Fix: Explicitly declare Callback API Version 1
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=f"pump_{BED_ID}")
client.username_pw_set("infusion-pump", PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_start()

while True:
    state["volume_remaining_ml"] -= (state["rate_ml_hr"] / 3600)
    payload = json.dumps({
        "rate": state["rate_ml_hr"],
        "volume_remaining": round(state["volume_remaining_ml"], 2)
    })
    
    # Fix: Route to ward1/
    client.publish(f"ward1/{BED_ID}/pump/status", payload)
    time.sleep(1)