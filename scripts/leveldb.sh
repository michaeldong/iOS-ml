#!/usr/bin/env sh
# Create the imagenet leveldb inputs
# N.B. set the path to the imagenet train + val data dirs

TOOLS=/opt/caffe/.build_release/tools
DATA=../images
leveldbDir=../leveldb

echo "Creating leveldb..."

rm -R $leveldbDir

mkdir $leveldbDir

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin -resize_width 256 -resize_height 256 -shuffle \
    "" \
    $DATA/train.txt \
    $leveldbDir/train_db

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin -resize_width 256 -resize_height 256 -shuffle \
    "" \
    $DATA/val.txt \
    $leveldbDir/val_db

echo "Done."
