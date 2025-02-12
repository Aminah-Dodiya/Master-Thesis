#!/bin/bash

# Start the server and redirect its output to server.log
python3 server.py > server.log 2>&1 &
sleep 3  # Sleep for 3s to give the server enough time to start

# Start the first client and redirect its output to client1.log
python3 client1.py > client1.log 2>&1 &

# Start the second client and redirect its output to client2.log
python3 client2.py > client2.log 2>&1 &

# Start the third client and redirect its output to client2.log
# python3 client3.py > client3.log 2>&1 &

# Start the fourth client and redirect its output to client2.log
# python3 client4.py > client4.log 2>&1 &