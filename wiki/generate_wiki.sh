#!/bin/bash
PLOTPALPATH=../
WIKIPATH=../../plotpal.wiki/
STARTDIR=$PWD

cd $PLOTPALPATH
pydoc-markdown -p plotpal > $STARTDIR/full_plotpal.md

cd $STARTDIR
python3 split_wiki.py $WIKIPATH
rm full_plotpal.md
