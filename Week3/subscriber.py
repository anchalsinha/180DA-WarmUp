import paho.mqtt.client as mqtt

# define callbacks
def on_connect(client, userdata, flags, rc):
    print("Connection returned result: "+str(rc))
    client.subscribe("ece180d/test", qos=1)

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


# connect to a broker using one of the connect*() functions.
client.connect_async('mqtt.eclipse.org')


# call one of the loop*() functions to maintain network traffic flow with the broker.
client.loop_start()
while True:
    pass


client.loop_stop()
client.disconnect()
