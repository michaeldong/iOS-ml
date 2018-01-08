TOOLS=/opt/caffe/build/tools
weightsDir=/opt/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel

GLOG_logtosterr=1 $TOOLS//caffe train -solver ../config/solver.prototxt -weights $weightsDir -gpu 0 >> output_finetune.txt
#GLOG_logtostderr=1  $TOOLS//caffe train -solver ../config/solver.prototxt -weights $weightsDir -gpu 0 >> output_finetune.txt 2>&1

echo "Done."
