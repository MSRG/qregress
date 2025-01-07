#!/bin/bash
pdflatex A1.tex
pdflatex A2.tex
pdflatex M.tex
pdflatex IQP.tex

convert -density 300 A1.pdf -quality 100 A1.png
convert -density 300 A2.pdf -quality 100 A2.png
convert -density 300 M.pdf -quality 100 M.png
convert -density 300 IQP.pdf -quality 100 IQP.png

