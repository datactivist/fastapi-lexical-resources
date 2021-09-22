#! /usr/bin/env bash

# docker path by default
path="/app/preprocess/"

while [ -n "$1" ]; do # while loop starts

	case "$1" in

	-l) path="preprocess/" ;; #if local mode, change path for local deployment

	*) echo "Option $1 not recognized" ;; # In case you typed a different option other than a,b,c

	esac

	shift

done


echo ""
echo "Starting conversion of embeddings to .magnitude, this may take some times..."
script="convert_embeddings.py"
full=$path$script
python3 $full
echo "Conversion done"
echo ""

echo ""
echo "Starting preload of embeddings using most_similar function"
script="preload_embeddings.py"
full=$path$script
python3 $full
echo "Preloading done"
echo ""

echo ""
echo "Starting saving of referentiels vectors"
script="preload_referentiels.py"
full=$path$script
python3 $full
echo "Saving done"
echo ""

