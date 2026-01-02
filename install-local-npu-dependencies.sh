#!/bin/bash

# Define the location where Radxa/Cix stores their wheels
WHEEL_DIR="/usr/share/cix/pypi"

echo "--- Checking for Cix NPU Drivers in $WHEEL_DIR ---"

if [ ! -d "$WHEEL_DIR" ]; then
    echo "ERROR: Directory $WHEEL_DIR not found!"
    echo "Are you running this on the official Radxa Orion O6 OS?"
    exit 1
fi

# Install libnoe (The Driver)
echo "Installing libnoe..."
find "$WHEEL_DIR" -name "libnoe*.whl" -exec pip install {} \;

# Install NOE_Engine (The Python Wrapper)
echo "Installing NOE_Engine..."
find "$WHEEL_DIR" -name "NOE_Engine*.whl" -exec pip install {} \;

# Install Operators (The Math)
echo "Installing ZhouyiOperators..."
find "$WHEEL_DIR" -name "ZhouyiOperators*.whl" -exec pip install {} \;

echo "--- Driver Installation Complete ---"
echo "Installing Python dependencies..."
pip install -r requirements.txt