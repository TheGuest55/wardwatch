import os

import paho.mqtt.client as mqtt


def build_client(client_id):
    """Returns a client that reconnects with exponential backoff."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    client.username_pw_set(os.environ["MQTT_USERNAME"], os.environ["MQTT_PASSWORD"])
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    return client


def connect(client):
    host = os.environ.get("MQTT_HOST", "172.28.10.10")
    port = int(os.environ.get("MQTT_PORT", "1883"))
    client.connect_async(host, port, keepalive=30)
