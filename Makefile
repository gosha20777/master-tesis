all: build clean run

build:
	latexmk -xelatex \
	-synctex=1 literature-overview.tex
	mv literature-overview.pdf build/literature-overview.pdf
	
run:
	evince main.pdf

docker:
	echo "master tesis in\n\t `pwd` directory"
	docker run \
		--rm \
		-v `pwd`:/master-tesis \
		docker-latex
	echo "done!"

clean:
	rm -f *.aux \
	*.fdb_latexmk \
	*.fls \
	*.log \
	*.out \
	*.synctex.gz \
	*.toc \
	*.xdv
