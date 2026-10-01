#!/bin/bash

for filename in $(ls -l ../Jupyter/ | grep .ipynb | awk '{print $9}');
do
        marimo convert $filename > ${filename:0:-6}.py
done
