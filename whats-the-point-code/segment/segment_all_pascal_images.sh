#!/bin/bash

for f in /imagenetdb3/abearman/data/PASCAL_ppm/*.ppm; do
	if [ ! -f /imagenetdb3/abearman/data/pascal-oversegmented/"${f##*/}" ]; then
		./segment 0.5 500 50 "$f" /imagenetdb3/abearman/data/pascal-oversegmented/"${f##*/}"
	fi
done

#for f in *.jpg; do convert ./"$f" -depth 8 ./"${f%.jpg}.ppm"; done

