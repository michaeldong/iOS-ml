 import os
 import sys
 import coremltools
 
 def main():
	   if len(sys.argv) < 2:
	   	 print len(sys.argv)
	   	 sys.exit(1)

	   # Convert a caffe model to a classifier in Core ML
	   caffemodelname = "snapshot_iter_" + sys.argv[1] + ".caffemodel"
    	print caffemodelname

	   coreml_model = coremltools.converters.caffe.convert((caffemodelname, 'deploy.prototxt', 'mean.binaryproto'), image_input_names = 'data', class_labels = 'labels.txt',is_bgr=True, image_scale=256.)

	# Now save the model
	   coreml_model.author = "author_name_XXXX"
	coreml_model.save('MMMM.mlmodel')


if __name__ == '__main__':
      main()
