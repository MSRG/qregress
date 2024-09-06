#!/bin/bash
dirs=$(find . -name "IQP_Full-Pauli-CRX.json")

new_name="IQP_Full-Pauli-CRZ"

for i in ${dirs}; do
 mv -f $i $(dirname $i)/${new_name}.json
done
