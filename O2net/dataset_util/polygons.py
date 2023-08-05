Hi @jianminsun, thanks for reporting the missing file. We will release it with the Cityscapes models and training instructions (please see issue #27 which tracks this).

In the meantime, you can obtain the instnaces2dict_with_polygons.py script by applying the following changes to the instances2dict.py script from the Cityscapes repo:

$ diff instances2dict.py instances2dict_with_polygons.py
--- instances2dict.py   2018-02-06 05:19:33.009812039 -0800
+++ instances2dict_with_polygons.py     2018-02-06 05:22:15.953031446 -0800
@@ -11,7 +11,9 @@
 sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
 from csHelpers import *

-def instances2dict(imageFileList, verbose=False):
+import cv2
+
+def instances2dict_with_polygons(imageFileList, verbose=False):
     imgCount     = 0
     instanceDict = {}

@@ -35,9 +37,22 @@

         # Loop through all instance ids in instance image
         for instanceId in np.unique(imgNp):
+            if instanceId < 1000:
+                continue
+
             instanceObj = Instance(imgNp, instanceId)
+            instanceObj_dict = instanceObj.toDict()

-            instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
+            if id2label[instanceObj.labelID].hasInstances:
+                mask = (imgNp == instanceId).astype(np.uint8)
+                contour, hier = cv2.findContours(
+                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
+
+                polygons = [c.reshape(-1).tolist() for c in contour]
+                instanceObj_dict['contours'] = polygons
+
+            instances[id2label[instanceObj.labelID].name].append(
+                instanceObj_dict)

         imgKey = os.path.abspath(imageFileName)
         instanceDict[imgKey] = instances





         ###################################

         #!/usr/bin/python
#
# Convert instances from png files to a dictionary
# This files is created according to https://github.com/facebookresearch/Detectron/issues/111

from __future__ import print_function, absolute_import, division
import os, sys

sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from csHelpers import *

# Cityscapes imports
from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *
import cv2
from maskrcnn_benchmark.utils import cv2_util


def instances2dict_with_polygons(imageFileList, verbose=False):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            #instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                contour, hier = cv2_util.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict['contours'] = polygons

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict_with_polygons(fileList, True)

if __name__ == "__main__":
    main(sys.argv[1:])



###########################################

# Setting Up Datasets
This file describes how to perform training on other datasets.

Only Pascal VOC dataset can be loaded from its original format and be outputted to Pascal style results currently.

We expect the annotations from other datasets be converted to COCO json format, and
the output will be in COCO-style. (i.e. AP, AP50, AP75, APs, APm, APl for bbox and segm)

## Creating Symlinks for PASCAL VOC

We assume that your symlinked `datasets/voc/VOC<year>` directory has the following structure:

```
VOC<year>
|_ JPEGImages
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ Annotations
|  |_ pascal_train<year>.json (optional)
|  |_ pascal_val<year>.json (optional)
|  |_ pascal_test<year>.json (optional)
|  |_ <im-1-name>.xml
|  |_ ...
|  |_ <im-N-name>.xml
|_ VOCdevkit<year>
```

Create symlinks for `voc/VOC<year>`:

```
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/voc/VOC<year>
ln -s /path/to/VOC<year> /datasets/voc/VOC<year>
```
Example configuration files for PASCAL VOC could be found [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/pascal_voc/).

### PASCAL VOC Annotations in COCO Format
To output COCO-style evaluation result, PASCAL VOC annotations in COCO json format is required and could be downloaded from [here](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip)
via http://cocodataset.org/#external.

## Creating Symlinks for Cityscapes:

We assume that your symlinked `datasets/cityscapes` directory has the following structure:

```
cityscapes
|_ images
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ annotations
|  |_ instanceonly_gtFile_train.json
|  |_ ...
|_ raw
   |_ gtFine
   |_ ...
   |_ README.md
```

Create symlinks for `cityscapes`:

```
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/cityscapes
ln -s /path/to/cityscapes datasets/data/cityscapes
```

### Steps to convert Cityscapes Annotations to COCO Format
1. Download gtFine_trainvaltest.zip from https://www.cityscapes-dataset.com/downloads/ (login required)
2. Extract it to /path/to/gtFine_trainvaltest
```
cityscapes
|_ gtFine_trainvaltest.zip
|_ gtFine_trainvaltest
   |_ gtFine
```
3. Run the below commands to convert the annotations

```
cd ~/github
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts
cp ~/github/maskrcnn-benchmark/tools/cityscapes/instances2dict_with_polygons.py cityscapesscripts/evaluation
python setup.py install
cd ~/github/maskrcnn-benchmark
python tools/cityscapes/convert_cityscapes_to_coco.py --datadir /path/to/cityscapes --outdir /path/to/cityscapes/annotations
```

Example configuration files for Cityscapes could be found [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/cityscapes/).