import paho.mqtt.client as mqtt
import numpy as np

# define callbacks
def on_connect(client, userdata, flags, rc):
    print("Connection returned result: "+str(rc))

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print('Unexpected Disconnect')
    else:
        print('Expected Disconnect')

def on_message(client, userdata, message):
    print('Received message: "' + str(message.payload) + '" on topic "' + message.topic + '" with QoS ' + str(message.qos))


# create a client instance.
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

# connect to a broker
client.connect_async('mqtt.eclipse.org')

# call one of the loop*() functions to maintain network traffic flow with the broker.
client.loop_start()

for i in range(10):
    client.publish('ece180d/test', float(np.random.random(1)), qos=1)


client.loop_stop()
client.disconnect()
