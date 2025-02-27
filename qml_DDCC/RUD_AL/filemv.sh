#!/bin/bash
dirs=$(find . -mindepth 1 -maxdepth 1 -type d -name "DDCC_*")
for dir in ./DDCC_*; do
    mv -f "$dir" "${dir#./DDCC_}"
done
