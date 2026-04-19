TEX_DIR := .
TEX_SRC := sbc-template.tex
OUT_BASENAME := B283069_ASPP_Report2
OUT_PDF := $(OUT_BASENAME).pdf

.PHONY: report clean deep-clean

report:
	cd $(TEX_DIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=$(OUT_BASENAME) $(TEX_SRC)

clean:
	cd $(TEX_DIR) && latexmk -C -jobname=$(OUT_BASENAME) $(TEX_SRC)
	rm -f $(OUT_PDF)

deep-clean: clean
	find $(TEX_DIR) -maxdepth 1 -type f \( \
		-name '*.aux' -o -name '*.bbl' -o -name '*.bcf' -o -name '*.blg' -o \
		-name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.log' -o -name '*.out' -o \
		-name '*.run.xml' -o -name '*.synctex.gz' -o -name '*.toc' \) -delete
