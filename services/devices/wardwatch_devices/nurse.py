from .client import build_client, connect


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"connected: {reason_code}", flush=True)
    if not reason_code.is_failure:
        client.subscribe("ward1/#", qos=0)


def on_message(client, userdata, msg):
    print(f"{msg.topic} {msg.payload.decode(errors='replace')}", flush=True)


def run():
    client = build_client("nurse-station")
    client.on_connect = on_connect
    client.on_message = on_message
    connect(client)
    client.loop_forever(retry_first_connection=True)
