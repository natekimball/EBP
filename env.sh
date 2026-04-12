#!/bin/bash

ENV_NAME="ENV"
REQUIREMENTS="requirements.txt"

if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"

    echo "Activating environment..."
    source "$ENV_NAME/bin/activate"

    pip install -r "$REQUIREMENTS"
    pip install flash-attn --no-build-isolation

    echo "Environment '$ENV_NAME' created and ready."
fi

source "$ENV_NAME/bin/activate"

echo "Activated environment: $VIRTUAL_ENV"
