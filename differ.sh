#!/bin/bash
#latexdiff --config="PICTUREENV=(?:picture|DIFnomarkup|align|tabular)[\w\d*@]*" paper.tex new_paper.tex > diff.tex
# latexdiff --exclude-textcmd=”section,subsection,subsubsection” --config=”PICTUREENV=(?:picture|DIFnomarkup|align|tabular)[\w\d*@]*” paper.tex new_paper.tex > diff.tex
# latexdiff paper.tex new_paper.tex > diff.tex
#latexmk -pdf -f diff
latexmk -pdf paper
latexmk -pdf new_paper
latexmk -pdf reply
