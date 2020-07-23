#! /bin/bash

#echo "Test"

LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH:/opt/app-root/src/
export LD_LIBRARY_PATH

LD_LIBRARY_PATH=./ python3 app.py

exit 0
