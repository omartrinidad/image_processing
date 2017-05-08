
filename=talk

pdf:
	cp -rf ../resulting_images/* images/
	pdflatex ${filename}

read:
	evince ${filename}.pdf &

add:
	git add Makefile
	git add chapters/
	git add notes.tex
	git add code
	git status

clean:
	 rm -f ${filename}.ps ${filename}.log ${filename}.aux
	 rm -f ${filename}.out ${filename}.dvi ${filename}.bbl ${filename}.blg
	 rm -f ${filename}.toc ${filename}.lof ${filename}.lot ${filename}.brf
	 rm -f ${filename}.pyg
	 rm -f ${filename}.nav
	 rm -f ${filename}.snm
	 rm -f texput.log
