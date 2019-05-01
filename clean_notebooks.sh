#!/bin/sh
# pre commit hook that clears the output of all jupyter notebooks so they dont clutter the repo
echo "clearing output of all jupyter notebooks in the notebooks dir"
files=$(ls notebooks/*.ipynb)
for file in $files
do
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $file
done
files=$(ls notebooks/experiments/*.ipynb)
for file in $files
do
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $file
done

echo "done"