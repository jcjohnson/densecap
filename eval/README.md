
# Dense Captioning mAP evaluation

Utility functions that compute the Average Precision Dense Captioning metric.

The Dense Captioning metric combines object detection with language into one metric.
- For object detection we compute IoU with ground truth.
- For language we compute [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/README.html) between prediction and true description.

Instead of sweeping over multiple IoU thresholds when comuting object detection mAP
we simultaneously sweep over both IoU and METEOR match thresholds in a 2D grid,
evaluating AP for each (IoU threshold, METEOR threshold). We then average over all
points in the 2D grid to get the final Dense Captioning mAP.

**Test**. The unit tests for these utilities is in `test/evaluation_test.lua`.

**Requirements**: 

- Java. METEOR requires Java installed. On Ubuntu you can do `sudo apt-get install default-jre`. You can do `java -version` to confirm that you have it installed.
- You'll need to download the METEOR binary `meteor-1.5.jar` and place it in `eval` folder and also `paraphrase-en.gz` and place it it in `eval/data` folder. Better instructions todo.

