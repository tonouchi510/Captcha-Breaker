#!/bin/sh
for i in {1..200000}
do
    convert -crop 100x100+57+0 out/train/1/$i.png out2/train/1/$i.png;
done