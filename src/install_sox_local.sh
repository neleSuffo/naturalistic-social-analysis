#!/bin/bash

# Variables
SOX_VERSION="14.4.2"
INSTALL_DIR="$HOME/.local"

# Create install directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Download and extract Sox
wget https://sourceforge.net/projects/sox/files/sox/${SOX_VERSION}/sox-${SOX_VERSION}.tar.gz -O sox-${SOX_VERSION}.tar.gz
tar -xzf sox-${SOX_VERSION}.tar.gz
cd sox-${SOX_VERSION}

# Install necessary dependencies locally
mkdir -p "$INSTALL_DIR/include" "$INSTALL_DIR/lib" "$INSTALL_DIR/bin"

# Build and install Sox
./configure --prefix=${INSTALL_DIR}
make
make install

# Add the installation directory to the PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Clean up
cd ..
rm -rf sox-${SOX_VERSION} sox-${SOX_VERSION}.tar.gz

echo "Sox installed successfully in $INSTALL_DIR"
