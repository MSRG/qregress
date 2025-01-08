#!/bin/bash
names=$(cat names.txt)

for i in $names; do
 if [ ! -f "$i.png" ]; then
  echo $i
  pdflatex $i.tex
  convert -density 300 $i.pdf -quality 100 $i.png
 fi
done
