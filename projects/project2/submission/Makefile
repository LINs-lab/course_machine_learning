# Makefile -- builds main.pdf via pdflatex + bibtex

MAIN = main

all:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).pdf $(MAIN).toc $(MAIN).fdb_latexmk $(MAIN).fls \
	      $(MAIN).synctex.gz sections/*.aux appendix/*.aux

.PHONY: all clean
