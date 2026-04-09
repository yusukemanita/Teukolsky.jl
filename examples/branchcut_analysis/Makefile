# Build targets for branchcut_note.tex
# Requires: latexmk, julia

.PHONY: figures pdf clean

figures:
	julia generate_figures.jl

pdf: figures
	latexmk -pdf branchcut_note.tex

clean:
	latexmk -C branchcut_note.tex
	rm -f *.pdf *.aux *.log *.toc *.out
