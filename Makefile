BASENAME = paper 

.PHONY: paper.pdf all clean

all: paper.pdf

# -pdf tells latexmk to generate PDF directly (instead of DVI).
# -pdflatex="" tells latexmk to call a specific backend with specific options.
# -use-make tells latexmk to call make for generating missing files.

# -interaction=nonstopmode keeps the pdflatex backend from stopping at a
# missing file reference and interactively asking you for an alternative.

paper.pdf: paper.tex data/
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make paper.tex

presentation.pdf: presentation.tex data/
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make presentation.tex

bbl:
	bibtex paper.aux
	bibtex paper.aux

clean:
	latexmk -CA
ps::
	latexmk -ps ${BASENAME}

# clean::
# 	latexmk -c -pdf ${BASENAME}

cleaner::
	rm -rvf *.dvi *.bbl *.pdf *.blg *.log *.aux *.out *.fls *.fdb_latexmk *~

cleanest:
	rm -rvf *.dvi *.bbl *.pdf *.blg *.log *.aux *.out *.fls *.fdb_latexmk *.png *.txt *~ *.log *.nav *.out *.snm *.toc 

diff:
	latexdiff paper_submitted.tex paper.tex > paper_diff.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make paper_diff.tex
