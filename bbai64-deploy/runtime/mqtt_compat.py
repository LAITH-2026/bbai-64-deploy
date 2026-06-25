"""paho-mqtt 1.x / 2.x compatibility.

paho-mqtt 2.0 made `callback_api_version` a required first constructor argument
and changed the callback signatures. The board's `pip install paho-mqtt` pulls
2.x, so `mqtt.Client()` (1.x style) raises there. `make_client()` builds a client
that keeps the classic v1 callback signatures — `on_connect(client, userdata,
flags, rc)` / `on_message(client, userdata, msg)` — on BOTH versions, so the rest
of the code needs no version branching.
"""
from __future__ import annotations


def make_client(client_id: str = ""):
    import paho.mqtt.client as mqtt
    try:                                   # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    except (AttributeError, TypeError):    # paho-mqtt 1.x
        return mqtt.Client(client_id=client_id)
