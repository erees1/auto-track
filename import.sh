#!/bin/bash

USAGE="Usage: ./import.sh filepath [tracker_path]"
if [ $# -eq 0 ]
  then
    echo "Error: No filepath provided"
    echo $USAGE
    exit 1
fi


if [ $# -eq 2 ] ; then
    tracker_path=$2
else
    # Save the tracker path here to save typing it out each time
    tracker_path="$HOME/OneDrive/Documents/Finances/Spending Tracker/2021_tracker.xlsx"
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PYTHON_ENV=$DIR/venv
source $PYTHON_ENV/bin/activate

# Run import script
python3 auto-track/src/import.py --csv_path "$1" --tracker_path "$tracker_path"
