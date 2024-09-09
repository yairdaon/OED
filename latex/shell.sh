latexmk -c
latexpand ba.tex > flat.tex
latexdiff --math-markup=OFF old/old_flat.tex flat.tex > diff.tex
latexmk --pdf ba.tex
latexmk --pdf diff.tex

