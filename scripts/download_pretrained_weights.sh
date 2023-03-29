#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/../log/train_all100k_data/
gdown https://drive.google.com/u/0/uc?id=1pz9eheQRQfx8itLj3nSKXQylTuG8DtB_&export=download
