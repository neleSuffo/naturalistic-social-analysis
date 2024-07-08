#!/usr/bin/bash

# Install Node.js
wget https://nodejs.org/dist/v20.11.0/node-v20.11.0-linux-x64.tar.xz &&\
mv node-v20.11.0-linux-x64.tar.xz ~/ &&\
cd ~ &&\
tar -xf ~/node-v20.11.0-linux-x64.tar.xz &&\
sudo ln -s ~/node-v20.11.0-linux-x64/bin/node /usr/bin/node &&\
sudo ln -s ~/node-v20.11.0-linux-x64/bin/npm /usr/bin/npm
sudo ln -s ~/node-v20.11.0-linux-x64/bin/nodemon /usr/bin/nodemon

# Install duckdb-async
~/node-v20.11.0-linux-x64/bin/npm install duckdb-async@0.10.2

# Install nodemon for watching changes in ipynb files
~/node-v20.11.0-linux-x64/bin/npm install -g nodemon

# Install duckdb CLI.
wget https://github.com/duckdb/duckdb/releases/download/v0.10.2/duckdb_cli-linux-amd64.zip &&\
unzip duckdb_cli-linux-amd64.zip &&\
sudo mv duckdb /usr/bin/

# Install libsecret
sudo apt update &&\
sudo apt install -y libsecret-1-0

# Add aliases to ~/.bash_aliases
echo "alias ll='ls -l'" >> ~/.bash_aliases
echo "alias la='ls -la'" >> ~/.bash_aliases