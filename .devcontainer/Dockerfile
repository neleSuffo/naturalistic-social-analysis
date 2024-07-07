# Use an official Python runtime as a parent image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.8-bullseye

# Install necessary packages
RUN apt-get update && apt-get install -y curl git bzip2 libgl1-mesa-glx




# The following code is for cloning the repository and setting up the Conda environment

# # Install Miniforge (a Conda installer) with detailed logging
# RUN curl -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
#     chmod +x /tmp/miniforge.sh && \
#     /tmp/miniforge.sh -b -p /opt/conda && \
#     rm /tmp/miniforge.sh

# # Set environment variables for Conda
# ENV PATH /opt/conda/bin:$PATH

# # Clone the voice_type_classifier repository and its submodules
# RUN git clone --recurse-submodules https://github.com/MarvinLvn/voice_type_classifier.git /workspace/voice_type_classifier

# # Create and activate the Conda environment
# RUN /opt/conda/bin/conda env create -f /workspace/voice_type_classifier/vtc.yml && \
#     /opt/conda/bin/conda clean -a -y && \
#     echo "conda activate pyannote" >> ~/.bashrc
