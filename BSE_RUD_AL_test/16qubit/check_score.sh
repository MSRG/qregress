#!/bin/bash
scores=$(find . -name "*results.json")

for i in $scores; do
  echo "$(dirname $i)"
  cat $i
  echo
done
