latexpand ba.tex > flat.tex
latexdiff --math-markup=OFF old.tex flat.tex > diff.tex
latexmk --pdf ba.tex
latexmk --pdf ba.tex
latexmk --pdf ba.tex
latexmk --pdf ba.tex
latexmk --pdf diff.tex
latexmk --pdf diff.tex
latexmk --pdf diff.tex
latexmk --pdf diff.tex
