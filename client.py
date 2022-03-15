import ubjson
import socket
import struct

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(("127.0.0.1", 1071))

    try:
        inp = 'read'
        if inp == 'capture':
            s.sendall(b'b')
            packet = s.recv(1024)
            while packet == b'':
                packet = s.recv(1024)

            print(struct.unpack('%sf' % 2, packet))
        elif inp == 'close':
            s.sendall(b'c')
        elif inp == 'read':
            s.sendall(b'r' + ubjson.dumpb({u'c': (50.24535, 53.21728), u'e': (50.245934, 53.217458)}))
            packet = s.recv(1024)
            print(ubjson.loadb(packet))
    finally:
        s.sendall(b'c')
