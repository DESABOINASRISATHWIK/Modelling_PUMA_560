# PUMA 560 — Group 7 Beamer presentation

Converted from `Group_7_PUMA560.pptx` supplied by the user. The 14-slide order,
content, embedded figures, and navy theme are retained. Text, equations, and
tables are editable LaTeX; the original figures remain raster images.
Typography and layout have been adapted to Beamer rather than reproduced pixel for pixel.
The demonstration slide retains the supplied deck's discussion of overlays.

## Contents

- `main.tex`: complete editable Beamer source
- `assets/`: original embedded PowerPoint figures
- `output/main.pdf`: compiled presentation
- `output/main.log`: compiler diagnostics
- `build.bat` / `build.sh`: build using a local TeX installation

## Build

Upload `main.tex` and `assets/` to Overleaf and choose pdfLaTeX, or run:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
```

With Tectonic installed, use:

```sh
tectonic main.tex --outdir output --keep-logs --keep-intermediates
```

Run commands from this directory. Tectonic downloads packages on its first build;
MiKTeX or TeX Live requires the standard Beamer, Latin Modern, and booktabs packages.
The supplied PDF was compiled with Tectonic 0.17.0.

Joint limits are reproduced from the supplied deck, in degrees.
The underlying project uses the same values in `src/robot_parameters.py`.
Slide numbers are consecutive in the conversion; the source deck had gaps.
