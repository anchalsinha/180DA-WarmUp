import socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.bind(('0.0.0.0', 8080))
server.listen(5)
while True:
    conn, addr = server.accept()
    recv_client = ''
    while True:
        data = conn.recv(4096)
        if not data: break
        recv_client += data
        print(recv_client)
        conn.send("Sending from the server\n")

    conn.close()
    print('Client disconnected')
