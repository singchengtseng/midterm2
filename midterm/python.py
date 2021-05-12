import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as paho
import time
import serial
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev, 9600)

path = 'gesture_ring.txt'
f=open(path,'r')
print(f.read())
f.close()

# https://os.mbed.com/teams/mqtt/wiki/Using-MQTT#python-client

# MQTT broker hosted on local machine
mqttc = paho.Client()

# Settings for connection
# TODO: revise host to your IP
host = "172.22.1.91"
topic = "Mbed"

Time=0
angle=[]
FLAG=1

# Callbacks
def on_connect(self, mosq, obj, rc):
    print("Connected rc: " + str(rc))

def on_message(mosq, obj, msg):
    global FLAG
    global Time
    global angle
    print("[Received] Topic: " + msg.topic + ", selected angle: " + str(msg.payload) + "\n")
    if FLAG==1:
        FLAG=FLAG+1
    else:
        Time=Time+1
        angle.append(str(msg.payload))
        if Time==10:
            print(angle)
    #s.write(bytes("/STOP/run {led1} {led2}\n".format(led1=x, led2=y), 'UTF-8'))

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
"""
# Publish messages from Python
num = 0
while num != 5:
    ret = mqttc.publish(topic, "Message from Python!\n", qos=0)
    if (ret[0] != 0):
            print("Publish failed")
    mqttc.loop()
    time.sleep(1.5)
    num += 1
"""
# Loop forever, receiving messages
mqttc.loop_forever()
