#!/usr/bin/env bash

set -x

python3 -m pip install -r requirements.txt

pushd ./data/poisson

python poisson.py # generate the dataset

popd

pushd src

python poisson/deeponet_poisson.py

popd