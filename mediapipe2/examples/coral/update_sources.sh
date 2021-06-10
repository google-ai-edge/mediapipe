#!/bin/bash

# To run in the Coral Docker environment.

. /etc/os-release

sed -i "s/deb\ /deb \[arch=amd64\]\ /g" /etc/apt/sources.list

echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME} main universe" >> /etc/apt/sources.list
echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME}-updates main universe" >> /etc/apt/sources.list
echo "deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME}-security main universe" >> /etc/apt/sources.list
