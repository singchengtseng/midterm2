import paho.mqtt.client as paho

import time

import serial


# https://os.mbed.com/teams/mqtt/wiki/Using-MQTT#python-client


# MQTT broker hosted on local machine

mqttc = paho.Client()


# Settings for connection

# TODO: revise host to your IP

host = "192.168.129.124"

topic = "Mbed"
angle = []


serdev = '/dev/ttyACM0'
s = serial.Serial(serdev, 9600)

# Callbacks

def on_connect(self, mosq, obj, rc):

    print("Connected rc: " + str(rc))


def on_message(mosq, obj, msg):
    print("[Received] Topic: " + msg.topic + ", Message: " + str(msg.payload) + "\n")
    angle.append(str(msg.payload))


def on_subscribe(mosq, obj, mid, granted_qos):

    print("Subscribed OK")


def on_unsubscribe(mosq, obj, mid, granted_qos):

    print("Unsubscribed OK")


# Set callbacks

mqttc.on_message = on_message

mqttc.on_connect = on_connect

mqttc.on_subscribe = on_subscribe

mqttc.on_unsubscribe = on_unsubscribe


# Connect and subscribe

print("Connecting to " + host + "/" + topic)

mqttc.connect(host, port=1883, keepalive=60)

mqttc.subscribe(topic, 0)


# Publish messages from Python
num = 0

if mqttc.on_message :
   num = num +1

if num == 10 :
   print(angle)
   s.write(bytes("/close2/run\r", 'UTF-8'))
    


# Loop forever, receiving messages

mqttc.loop_forever()