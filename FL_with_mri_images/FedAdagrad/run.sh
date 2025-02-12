#!/bin/bash

echo "Starting server"
# Start the server and redirect its output to server.log
python3 server.py > server.log 2>&1 &
sleep 3  # Sleep for 3s to give the server enough time to start

echo "Starting client1"
# Start the first client and redirect its output to client1.log
python3 client1.py > client1.log 2>&1 &

echo "starting client2"
# Start the second client and redirect its output to client2.log
python3 client2.py > client2.log 2>&1 &

echo "starting client3"
python3 client3.py > client3.log 2>&1 &

echo "Starting client4"
python3 client4.py > client4.log 2>&1 &

# this will allow you to use CTRL+c to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# wait for all background processes to complete
wait
