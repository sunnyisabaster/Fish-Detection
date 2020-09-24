# transform as coco data
import os
import pickle
import os
import numpy as np
import json
import re
import cv2
import glob
import shutil
import sys, stat
import pycocotools.mask as maskUtils



def get_fish_dicts():
    train_file = '/data/project/annotations/niap_fix_maskrcnn_train0_30.pik'
    val_file = '/data/project/annotations/niap_fix_maskrcnn_val_30_40.pik'
    test_file = '/data/project/annotations/niap_fix_maskrcnn_test_40_45.pik'
    with open(train_file, 'rb') as f:# ------------change here
        train_data = pickle.load(f)
    total_obj = max(train_data.keys())+1 # objs amount
    # total_obj = 10
    i = -1
    # dataset_dicts = []
    # d = 0
    open_train = '/data/mmdetection/data/coco/annotations/instances_train2017.json'
    open_val = '/data/mmdetection/data/coco/annotations/instances_val2017.json'
    open_test = '/data/mmdetection/data/coco/annotations/instances_test2017.json'
    f=open(open_train,'w')# ------------change here

    json_file = {
        # 'info': 'spytensor created',
        # 'license': ['license'],
        'annotations': [],
        'images': [],
        'categories':[{'id':0, 'name':'fish'}]
    }

    for v in range(total_obj):
        # i = i+1
        # record = {}
        # image details
        file_path = list(list(train_data[v].items())[1])[1]
        # print(file_path)
        filename = file_path.replace('/data/NIAP/Data/frames','/data/project/frame')

        height, width = cv2.imread(filename).shape[:2]
        really_filename = file_path.replace('/data/NIAP/Data/frames/', '')
        # find all image pathes and copy them to dataset folders
        # images_path = re.sub(r"/+\d+.jpg",'',really_filename)

        # absolute_path = os.path.join('/data/project/frame',images_path)

        # copy_dist = '/data/mmdetection/data/coco/train2017' + images_path

        # print(copy_dist)
        # if not os.path.exists(copy_dist):
        #    os.mkdir(copy_dist)
        #    os.chmod(copy_dist , stat.S_IRWXU|stat.S_IRGRP|stat.S_IROTH)
        #    shutil.copy( filename, copy_dist)
        # else:
        #    os.chmod(copy_dist , stat.S_IRWXU|stat.S_IRGRP|stat.S_IROTH)
        #    shutil.copy( filename, copy_dist)
        annos = train_data[v]['points']
        # print(annos)
        for point in annos:
        #annotation points
            # points = list(list(train_data[v].items())[0])[1]
            str_a = ','.join(str(x) for x in point)
            str_a1 = str_a.replace('[','')
            str_a2 = str_a1.replace(']','')
            clean_str = ' '.join(str_a2.split())
            str_a3 = re.sub('[\s+]',',',clean_str)
            points_list = str_a3.split(',')
            points_list = filter(None, points_list)#remove ''
            a_float_m = map(float, points_list)
            # print('::',points_list,'::')
            a_float_m = list(a_float_m)

            px = a_float_m[::2]
            py = a_float_m[1::2]


            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            # print(":",len(poly),":")

            if len(poly) <= 4:
                #print(v)
                #print(poly)
                #print(len(poly))
                #print("------------------break-----------------")
                continue
            # if len(poly) <= 3 and len(poly)>1:
            #    anno = {
            #        'segmentation':[np.min(px), np.min(py), np.max(px), np.max(py)],
            #        'area': 1.0,
            #        'iscrowd': 0,
            #        'image_id': v,
            #        'bbox':[np.min(px)+0.5, np.min(py)+0.5, np.max(px)+0.5, np.max(py)+0.5],
            #        'category_id': 0,
            #        'id': i #was d
            #    }
            #    print("------------------break-----------------")
            #    print(anno)
            #    print(len(poly))
            # elif len(poly) ==1 or len(poly) == 0:
            #   print("----------------111000-------------------")
            #   continue
            else:
                # calculate area
                rles = maskUtils.frPyObjects([poly], height, width)
                rle = maskUtils.merge(rles)
                area = float(maskUtils.area(rle))
                #area_ori = str(area)
                #area_1 = area_ori.replace('[','')
                #area_2 = area_1.replace(']','')
                #area_final = float(area_2)
                # print("area is:")
                # print(area)
                i = i+1
                anno = {
                    'segmentation':[poly],
                    'area': area,
                    'iscrowd': 0,
                    'image_id': v,
                    'bbox':[np.min(px), np.min(py), np.max(px), np.max(py)],
                    'category_id': 0,
                    'id': i #was d
                }
                # print("area:")
                # print(area)
            json_file['annotations'].append(anno)
                #json_file['images'].append(img)

                # record['info'] = 'spytensor created'
                # record['license'] = ['license']
                # record['images'] = img
                # record['annotations'] = anno
                # record['categories'] = cate
                # dataset_dicts.append(record)
                # str_a = ','.join(str(x) for x in dataset_dicts)
            # print('write_down:',i,'obj')
        img = {
            'file_name':really_filename,
            'height': height,
            'width': width,
            'id': v
        }
        json_file['images'].append(img)
    str_a = str(json_file)
    print(len(str_a))
    f.write(str_a)
    # break
    f.close()
    return json_file

json_file = get_fish_dicts()

# print(dataset_dicts)

import os
def alter(file,old_str,new_str):
    with open(file, "r") as f1,open("%s.bak" % file, "w") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)


alter("/data/mmdetection/data/coco/annotations/instances_train2017.json", "'", '"')
# ------------change here
print('Finished!')
