run command: python tools/train.py ./configs/detectors/htc_r50_sac_1x_coco.py
python tools/train.py ./configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py

cascade_rcnn_x101_32x4d_fpn_20e_coco.py

cascade_mask_rcnn_x101_32x4d_fpn_20e_coco.py

cascade_rcnn_x101_64x4d_fpn_20e_coco

./configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py

htc_without_semantic_r50_fpn_1x_coco.py change num_classes

/data/mmdetection/mmdet/core/evaluation/class_names.py changed, class name


/data/mmdetection/data/coco/annotations/instances_test.py change single qoute as double

:1,$ s/^/[/g   #add '[' at firts line
:1,$ s/$/]/g   #add ']' at last line

python tools/train.py ./configs/detectors/htc_r50_sac_1x_coco.py

坑：json文件''变成""
头尾加[ 和 ]？

/data/mmdetection/configs/htc$ vim htc_r50_fpn_1x_coco.py changed

/data/mmdetection/mmdet/datasets/custom.py commanded # if self.custom_classes:
                                                     # self.data_infos = self.get_subset_by_classes()
                                                     


#def coco_classes():
   # return [
   #     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
   #     'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
   #     'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
   #     'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
   #     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
   #     'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
   #     'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
   #     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
   #     'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
   #     'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
   #     'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
   #     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
   #     'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
   # ]
   
#CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    
    
nvidia-smi : check the GPU info
$ ps a
$ kill -s 9 1827

top : check momery

RuntimeError: DataLoader worker (pid 8881) is killed by signal: Terminated. -------
修改 /data/mmdetection/configs/_base_/datasets/coco_my.py workers_per_gpu=0,# was 2


python setup.py build_ext --inplace
export PYTHONPATH=${PWD}:$PYTHONPATH

训练时多出了segm，需要将_base_/dataset/coco_my.py 删掉evaluation的matric一个

rm -r file/ 删除文件夹和其下所有文件

/data/mmdetection/mmdet/datasets/pipelines/loading.py

annotation文件中有些bad点， 即为segmentation只有2个点或三个点，导致File "pycocotools/_mask.pyx", line 307, in pycocotools._mask.frPyObjects：Exception: input type is not supported.
check ur annotation label, there are some points' number smaller than 4

train:8985 write_down: 79895 objfish 79831
vla:3409    29524
test:1552   9237

File "<array_function internals>", line 6, in concatenate
ValueError: need at least one array to concatenate
need to check /data/mmdetection/mmdet/datasets/coco.py load_annotations annos is empty or not

test command: python tools/test.py ./configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py ./work_dirs/cascade_rcnn_x101_64x4d_fpn_20e_coco/latest.pth --out ./result.pkl

scp /data/mmdetection/test_hat_result/aja-helo-1H000314_2019-10-14_1002/0.jpg sunj5@student.computing.dcu.ie:bin/


python tools/robustness_eval.py ./result.pkl --dataset coco --metric AP #test AP


Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.001
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.002
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.002
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000

checked annotation id, could be similar to your image_id
check your load_form in the config file
优化器设置,使用SGD，学习率为：0.02/16*1=0.00125，其中1为我们使用一个GPU训练 16 = batch_size = 8*2

---------------------------------------------------------------------------------------------
预训练(pre-training/trained)与微调(fine-tuning)
什么是预训练和微调？

预训练(pre-training/trained)：你需要搭建一个网络来完成一个特定的图像分类的任务。首先，你需要随机初始化参数，然后开始训练网络，不断调整直到网络的损失越来越小。在训练的过程中，一开始初始化的参数会不断变化。当你觉得结果很满意的时候，就可以将训练模型的参数保存下来，以便训练好的模型可以在下次执行类似任务时获得较好的结果。这个过程就是pre-training。

之后，你又接收到一个类似的图像分类的任务。这个时候，你可以直接使用之前保存下来的模型的参数来作为这一任务的初始化参数，然后在训练的过程中，依据结果不断进行一些修改。这时候，你使用的就是一个pre-trained模型，而过程就是fine-tuning。

所以，预训练就是指预先训练的一个模型或者指预先训练模型的过程；微调 就是指将预训练过的模型作用于自己的数据集，并参数适应自己数据集的过程。

微调的作用

在CNN领域中。很少人自己从头训练一个CNN网络。主要原因上自己很小的概率会拥有足够大的数据集，从头训练，很容易造成过拟合。

所以，一般的操作都是在一个大型的数据集上训练一个模型，然后使用该模型作为类似任务的初始化或者特征提取器。比如VGG，Inception等模型都提供了自己的训练参数，以便人们可以拿来微调。这样既节省了时间和计算资源，又能很快的达到较好的效果。

什么是/为什么要迁移学习？
迁移学习(Transfer learning) 顾名思义就是就是把已学训练好的模型参数迁移到新的模型来帮助新模型训练。考虑到大部分数据或任务是存在相关性的，所以通过迁移学习我们可以将已经学到的模型参数（也可理解为模型学到的知识）通过某种方式来分享给新模型从而加快并优化模型的学习效率不用像大多数网络那样从零学习（starting from scratch，tabula rasa）。
---------------------------------------------------------------------------------------------
https://zhuanlan.zhihu.com/p/70878433：
cocodataset should like: 

'images': [
    {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },{
        'segmentation': [[88,
            33.09,
            ...
            444.03,
            555.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 0,
        'id': 42986
    }
    ...
],

'categories': [
    {'id': 0, 'name': 'fish'},
 ]
 
 
 TypeError: Argument ‘bb’ has incorrect type (expected numpy.ndarray, got list):自己的解决方法就是你同一条直线上加点，记得要构面，补齐4个以上就行了（这么做是我自己因为没有segmentation数据，我把bbox里面的数据拿来凑了4个点后就不报这个错误了）,但其实4个点也会报错 因为它4个点会认为是bbox格式而不是poly格式的数据。
 
 sample_per_gpu: 每个gpu的batch size
 workers_per_gpu: 每个gpu分配的线程数
 
                # calculate area
                rles = maskUtils.frPyObjects([poly], height, width)
                area = maskUtils.area(rles)
                area_ori = str(area) 
                area_1 = area_ori.replace('[','')
                area_2 = area_1.replace(']','')
                area_final = float(area_2)
                print("area is:")
                print(area_final)
              
weight size需要修改吗？：                
size mismatch for roi_head.bbox_head.0.fc_cls.weight: copying a param with shape torch.Size([1, 1024]) from checkpoint, the shape in current model is torch.Size([2, 1024]).

size mismatch for roi_head.bbox_head.0.fc_cls.bias: copying a param with shape torch.Size([1]) from checkpoint, the shape in current model is torch.Size([2]).

size mismatch for roi_head.bbox_head.1.fc_cls.weight: copying a param with shape torch.Size([1, 1024]) from checkpoint, the shape in current model is torch.Size([2, 1024]).

size mismatch for roi_head.bbox_head.1.fc_cls.bias: copying a param with shape torch.Size([1]) from checkpoint, the shape in current model is torch.Size([2]).

size mismatch for roi_head.bbox_head.2.fc_cls.weight: copying a param with shape torch.Size([1, 1024]) from checkpoint, the shape in current model is torch.Size([2, 1024]).

size mismatch for roi_head.bbox_head.2.fc_cls.bias: copying a param with shape torch.Size([1]) from checkpoint, the shape in current model is torch.Size([2]).


指定显卡：CUDA_VISIBLE_DEVICES=1 python tools/train.py ./configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco.py

htc_r50_sac_1x_coco.py

size mismatch for 
roi_head.mask_head.0.conv_logits.weight: copying a param with shape torch.Size([80, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([1, 256, 1, 1]).

size mismatch for roi_head.mask_head.0.conv_logits.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([1]).

size mismatch for roi_head.mask_head.1.conv_logits.weight: copying a param with shape torch.Size([80, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([1, 256, 1, 1]).

size mismatch for roi_head.mask_head.1.conv_logits.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([1]).

size mismatch for roi_head.mask_head.2.conv_logits.weight: copying a param with shape torch.Size([80, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([1, 256, 1, 1]).

size mismatch for roi_head.mask_head.2.conv_logits.bias: copying a param with shape torch.Size([80]) from checkpoint, the shape in current model is torch.Size([1]).

detectors固然好但是需要coco-suff数据 这个是我们没有的：HTC requires COCO and COCO-stuff dataset for training. You need to download and extract it in the COCO dataset path. The directory should be like this. https://www.cnblogs.com/nightmoonzjm/p/cocostuff_translate.html

目标检测】基础知识：IoU、NMS、Bounding box regression：https://zhuanlan.zhihu.com/p/64420287

/opt/conda/lib/python3.7/site-packages

1	import a_module
2	print a_module.__file__
上述代码将范围 .pyc 文件被加载的路径


CUDA_VISIBLE_DEVICES=0 

mask_rcnn_x101_64x4d_fpn_1x_coco.py


size mismatch for 
roi_head.bbox_head.0.fc_cls.weight: copying a param with shape torch.Size([81, 1024]) from checkpoint, the shape in current model is torch.Size([2, 1024]).
size mismatch for roi_head.bbox_head.0.fc_cls.bias: copying a param with shape torch.Size([81]) from checkpoint, the shape in current model is torch.Size([2]).
size mismatch for roi_head.bbox_head.1.fc_cls.weight: copying a param with shape torch.Size([81, 1024]) from checkpoint, the shape in current model is torch.Size([2, 1024]).
size mismatch for roi_head.bbox_head.1.fc_cls.bias: copying a param with shape torch.Size([81]) from checkpoint, the shape in current model is torch.Size([2]).
size mismatch for roi_head.bbox_head.2.fc_cls.weight: copying a param with shape torch.Size([81, 1024]) from checkpoint, the shape in current model is torch.Size([2, 1024]).
size mismatch for roi_head.bbox_head.2.fc_cls.bias: copying a param with shape torch.Size([81]) from checkpoint, the shape in current model is torch.Size([2]).

./darknet detector train cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show -map
./darknet detector train cfg/coco.data cfg/yolov4.cfg  backup/yolov4_last.weights -dont_show -map

 turminal2 need to 'source .bashrc' under /home/sun
 
 cascade_mask_rcnn_R_50_FPN_1x_72_59_model_fianl.pth
 
 CUDA_VISIBLE_DEVICES=1 python detectron2_project.py TjpT@FR3
