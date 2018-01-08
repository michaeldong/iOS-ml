leveldbDir=../leveldb
DATA=../mean
TOOLS=/opt/caffe/.build_release/tools
rm -R $DATA
mkdir $DATA
$TOOLS/compute_image_mean $leveldbDir/train_db \
  $DATA/mean.binaryproto

echo "Done."
