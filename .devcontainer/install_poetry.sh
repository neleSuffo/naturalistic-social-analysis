#!/usr/bin/bash

# Description: Install poetry
curl -SSL https://install.python-poetry.org | python3 - --version 1.8.2

# Install Poetry dependencies
poetry install --with dev --sync