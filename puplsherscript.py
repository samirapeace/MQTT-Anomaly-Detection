import paho.mqtt.client as mqtt
import time

# عنوان الوسيط (Mosquitto Broker)
BROKER = "192.168.117.1"
PORT = 1883

# إنشاء عميل MQTT
client = mqtt.Client()

# الاتصال بالوسيط
client.connect(BROKER, PORT, 60)

# تشغيل حلقة الاتصال في الخلفية
client.loop_start()

# مواضيع تمثل بيانات أجهزة IoT طبيعية
topics = [
    "iot/temperature",
    "iot/humidity",
    "iot/light"
]

# حلقة إرسال مستمرة (سلوك طبيعي)
while True:
    for topic in topics:
        # توليد Payload طبيعي (قيمة زمنية)
        payload = "value:" + str(time.time())

        # نشر الرسالة
        client.publish(topic, payload, qos=1)

        # انتظار قبل الإرسال التالي (سلوك طبيعي)
        time.sleep(2)
