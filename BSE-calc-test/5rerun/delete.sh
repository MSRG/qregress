#!/bin/bash

for i in ./finished/*; do
	rm -rf $(basename $i) 
done
