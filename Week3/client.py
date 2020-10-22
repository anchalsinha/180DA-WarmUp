import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('0.0.0.0', 8080))
client.send("This is the client\n")
recv_server = client.recv(4096)
client.close()
print(recv_server)
