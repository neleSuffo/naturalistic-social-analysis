# Use an official Python runtime as a parent image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.8-bullseye

# Install necessary packages
RUN apt-get update && apt-get install -y curl git bzip2 libgl1-mesa-glx

# Ensure the workspace directory has the correct permissions
RUN mkdir -p /workspaces && \
    chown -R vscode:vscode /workspaces && \
    chmod -R 755 /workspaces