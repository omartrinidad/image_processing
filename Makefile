
PUBLISH=slides3/

# all: compile bibliography recompile open
all: compile

clean:
	rm -rf $(PUBLISH)*.log $(PUBLISH)*.toc $(PUBLISH)*.toc $(PUBLISH)*.out
	rm -rf $(PUBLISH)*.nav $(PUBLISH)*.aux $(PUBLISH)*.snm

compile:
	pdflatex -output-directory $(PUBLISH) slides3.tex

#bibliography:
#	bibtex publish/gutierrez_cv

open:
	evince $(PUBLISH)slides3.pdf &
