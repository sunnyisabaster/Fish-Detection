# DemystifyingTheOceanThroughUnderwaterVideoAnalysis
For ocean environment protection, the government invested substantial time and effort in environmental protection, such as funding environment research and introducing new laws and regulations. Nowadays, artificial intelligence is considered an exact method because it may bring reforms to this area. Also, AI has acquired significant achievements, such as computer vision and deep learning fields. However, there are few publicly available, large scale underwater benchmark datasets. In this task, we first create such a dataset using videos captured by the SmartBay Ocean Observatory in Galway. Besides, a state-of-the-art deep model (Cascade Mask R-CNN) is re-trained using the benchmark dataset. The model achieved 72% mean average precision (mAP) on object detection (bounding box) and 66% mAP on the instance segmentation. Results show that the dataset created can provide sufficient information to train a deep model. Also, state-of-the-art deep models can detect and segment marine life with relatively high accuracy.

## Dataset:
Due to the limited space on Github, the annotated subset of videos (45 videos, ~3G) is hosted on Google Drive [link](https://drive.google.com/file/d/1SnWTu-3tgarfKXuq4vHjcjJZvHDUfi78/view?usp=sharing).

## Method
The architecture of Cascade R-CNN has shown in Figure 5, which is the multi-stage extension of Faster R-CNN. During the training, the hypotheses' quality and detector simultaneously could be increased through the extension part. A series of end-to-end trained indicators consist of the network. Because IoU thresholds increase continuously, sensors are more selective to false positives. At the interface, the objects can be bound by a bounding box, and give the label and accuracy. During the training, the Mask R-CNN is extended through parallelly adding the existing detection branch and segmentation branch in the Faster R-CNN. The structure of Mask R-CNN has been shown in Figure 7. Compare with Faster R-CNN, and it only has one more segmentation branch. Similar to the relationship between the Faster R-CNN and Mask R-CNN, the instance segmentation layers are the most significant difference.

## Tools
Some tools, e.g. extract frames from video, are available.

## Sample
TODO
- [x] add a Cascade Mask R-CNN training sample
- [ ] add a demo to how to use existing model
- [ ] add sample results here
- [ ] add sample output video here

!Result(https://www.bilibili.com/video/BV19k4y117gb)
