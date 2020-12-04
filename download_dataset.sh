#!/bin/bash

set -exu

# Download LJ Speech dataset using wget and extract
mkdir -p ./dataset
pushd ./dataset
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf ./dataset/LJSpeech-1.1.tar.bz2
popd

