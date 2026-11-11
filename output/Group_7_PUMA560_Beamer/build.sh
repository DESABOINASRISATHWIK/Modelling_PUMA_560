#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
mkdir -p output
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
