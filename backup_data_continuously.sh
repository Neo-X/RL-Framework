#!/bin/bash
while true
do
	echo "Backing up data"
	./backup_data.sh
    sleep 3600 ### 10 minutes
done
