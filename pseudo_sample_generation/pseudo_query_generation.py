import os
import time
import torch
import random
random.seed(20211024) # original_seed
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 9)  # small images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
from imageio import imread

conf_thresh = 0.4
MIN_BOXES = 10
MAX_BOXES = 36

predefined_hat_cls = []
predefined_shirt_cls = []
predefined_pants_cls = []
predefined_shoes_cls = []
predefined_people_cls = []
predefined_things_cls = []

with open(
        '../data/statistic/detect_hat_classes.txt',
        'r') as f:
    for line in f:
        predefined_hat_cls.append(line[:-1])

with open(
        '../data/statistic/detect_shirt_classes.txt',
        'r') as f:
    for line in f:
        predefined_shirt_cls.append(line[:-1])

with open(
        '../data/statistic/detect_pants_classes.txt',
        'r') as f:
    for line in f:
        predefined_pants_cls.append(line[:-1])

with open(
        '../data/statistic/detect_shoes_classes.txt',
        'r') as f:
    for line in f:
        predefined_shoes_cls.append(line[:-1])

predefined_cloth_cls = predefined_hat_cls + predefined_pants_cls + predefined_shirt_cls + predefined_shoes_cls

with open(
        '../data/statistic/detect_people_classes.txt',
        'r') as f:
    for line in f:
        predefined_people_cls.append(line[:-1])

with open(
        '../data/statistic/visual_genome_detect_classes.txt',
        'r') as f:
    for line in f:
        if line[:-1] not in predefined_cloth_cls + predefined_people_cls:
            predefined_things_cls.append(line[:-1])

predefined_cls = predefined_people_cls + predefined_cloth_cls + predefined_things_cls
predefined_cls = predefined_cls[1:]  # remove '__background__' class

bua_data_path = '../data/statistic/'
bua_attributes = ['__no_attribute__']
with open(os.path.join(bua_data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        bua_attributes.append(att.split(',')[0].lower().strip())


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--vg_dataset_path', dest='vg_dataset_path',
                        help='unlabeled dataset path',
                        default='/home/data/referit_data/', type=str)
    parser.add_argument('--vg_dataset', dest='vg_dataset',
                        help='unlabeled dataset',
                        default='vg', type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--image_out_dir', dest='image_out_dir',
                        help='directory to load images for demo',
                        default=None)
    parser.add_argument('--image_file', dest='image_file',
                        help='the file name of load images for demo',
                        default="img1.jpg")
    parser.add_argument('--image_list_file', dest='image_list_file',
                        help='the file name of load images for demo',
                        default="img1.jpg")
    parser.add_argument('--detection_file', dest='detection_file',
                        help='the file name of load images for demo',
                        default="img1.jpg")
    parser.add_argument('--attr_detection_file', dest='attr_detection_file',
                        help='the file name of load images for demo',
                        default="img1.jpg")
    parser.add_argument('--split_ind', dest='split_ind',
                        default=0, type=int)
    parser.add_argument('--each_image_query', dest='each_image_query',
                        default=10, type=int)
    parser.add_argument('--topn', dest='topn',
                        default=3, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--out_path', dest='out_path',
                        help='the file name of load images for demo',
                        default=None)
    parser.add_argument('--attr_iou_thresh', dest='attr_iou_thresh',
                        default=0.5, type=float)
    parser.add_argument('--attr_conf_thresh', dest='attr_conf_thresh',
                        default=0.3, type=float)
    args = parser.parse_args()
    return args


def filter_detect_cls(cls_name):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Filter out undesired class
    """
    return True


def is_cls_a_people(cls_name):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Check whether the class_name is kind of human being
    """
    if cls_name in predefined_people_cls:
        return True
    else:
        return False


def is_cls_a_cloth(cls_name):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Check whether the class_name is kind of human being
    """
    if cls_name in predefined_cloth_cls:
        return True
    else:
        return False


def is_hat_belong_to_people(hat_bbox, people_bbox, thresh=0.75):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Check the center_y of hat is above a threshold ratio of people height. Hat is always on head!
    """
    hat_y_center = (hat_bbox[3] + hat_bbox[1]) / 2
    people_height = people_bbox[3] - people_bbox[1]
    if (people_bbox[3] - hat_y_center) / people_height > thresh:
        return True
    else:
        return False


def is_shirt_belong_to_people():
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Check the shirt is belong to a people?
    """
    return True


def is_pants_belong_to_people(pants_bbox, people_bbox, thresh=0.5):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Check the center_y of hat is above a threshold ratio of people height. Hat is always on head!
    """
    pants_y_center = (pants_bbox[3] + pants_bbox[1]) / 2
    people_height = people_bbox[3] - people_bbox[1]
    if (people_bbox[3] - pants_y_center) / people_height < thresh:
        return True
    else:
        return False


def is_shoes_belong_to_people():
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Check the shoes is belong to a people?
    """
    return True


def center_of_bbox(bbox):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: center coordinate of bbox
    """
    return [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]


def area_of_bbox(bbox):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Area of a bbox
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def overlap_area_of_two_bbox(bbox1, bbox2):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Intersection of two bbox
    """
    xmin, xmax = max(bbox1[0], bbox2[0]), min(bbox1[2], bbox2[2])
    ymin, ymax = max(bbox1[1], bbox2[1]), min(bbox1[3], bbox2[3])
    width = xmax - xmin
    height = ymax - ymin
    if width < 0 or height < 0:
        return 0
    else:
        return width * height


def IoU(bbox1, bbox2):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: Intersection of Union of two bbox
    """
    overlap_area = overlap_area_of_two_bbox(bbox1, bbox2)
    if overlap_area == 0:
        return 0
    else:
        area_bbox1 = area_of_bbox(bbox1)
        area_bbox2 = area_of_bbox(bbox2)
        return overlap_area / (area_bbox1 + area_bbox2 - overlap_area)


def percent_of_bbox1_in_bbox2(bbox1, bbox2):
    """
        Create by Haojun Jiang, 2021/10/17
        Function: How many percent of area of bbox1 inside bbox2
    """
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    return overlap_area_of_two_bbox(bbox1, bbox2) / bbox1_area


def remove_overlap_bbox(descriptor, iou_thresh=0.4, reform2dict=True):
    tmp_descriptor = descriptor.copy()
    pop_ind = []
    for ind1 in range(len(descriptor) - 1):
        for ind2 in range(ind1 + 1, len(descriptor)):
            iou = IoU(descriptor[ind1]['bbox'], descriptor[ind2]['bbox'])
            if iou > iou_thresh:
                if descriptor[ind1]['bbox'][4] < descriptor[ind2]['bbox'][4]:  # keep the one with large confidence
                    if ind1 not in pop_ind:
                        pop_ind.append(ind1)
                else:
                    if ind2 not in pop_ind:
                        pop_ind.append(ind2)

    pop_ind = sorted(pop_ind, reverse=True)
    for ind in pop_ind:
        tmp_descriptor.pop(ind)

    if reform2dict:
        new_descriptor = {}
        for ind in range(len(tmp_descriptor)):
            people_info = tmp_descriptor[ind]
            cls = people_info['class']
            if cls not in new_descriptor:
                new_descriptor[cls] = []
            new_descriptor[cls].append(people_info)

        return new_descriptor
    else:
        return tmp_descriptor


def remove_tiny_bbox(descriptor, image_size, area_thresh=0.015625):
    tmp_descriptor = descriptor.copy()
    pop_ind = []
    image_area = image_size[0] * image_size[1]
    for ind in range(len(descriptor)):
        bbox_area = area_of_bbox(descriptor[ind]['bbox'])
        if bbox_area / image_area < area_thresh:
            pop_ind.append(ind)

    pop_ind = sorted(pop_ind, reverse=True)
    for ind in pop_ind:
        tmp_descriptor.pop(ind)

    return tmp_descriptor


def match_clothes_to_people(people_descriptor, clothes_descriptor, area_thresh=0.75):
    """
        Create by Haojun Jiang, 2021/10/17
        Function:
            1.Match the clothes to people, and remove the clothes that can not find a matching people
    """
    for cloth_info in clothes_descriptor:
        for key, value in people_descriptor.items():
            for ind, people_info in enumerate(value):
                if percent_of_bbox1_in_bbox2(cloth_info['bbox'], people_info['bbox']) > area_thresh:
                    if cloth_info['class'] in predefined_hat_cls:
                        if is_hat_belong_to_people(cloth_info['bbox'], people_info['bbox']):
                            if people_descriptor[key][ind]['clothes']['hat'] is None:
                                people_descriptor[key][ind]['clothes']['hat'] = []
                            people_descriptor[key][ind]['clothes']['hat'].append(cloth_info)
                        else:
                            pass
                    elif cloth_info['class'] in predefined_shirt_cls:
                        if is_shirt_belong_to_people():
                            if people_descriptor[key][ind]['clothes']['shirt'] is None:
                                people_descriptor[key][ind]['clothes']['shirt'] = []
                            people_descriptor[key][ind]['clothes']['shirt'].append(cloth_info)
                        else:
                            pass
                    elif cloth_info['class'] in predefined_pants_cls:
                        if is_pants_belong_to_people(cloth_info['bbox'], people_info['bbox']):
                            if people_descriptor[key][ind]['clothes']['pants'] is None:
                                people_descriptor[key][ind]['clothes']['pants'] = []
                            people_descriptor[key][ind]['clothes']['pants'].append(cloth_info)
                        else:
                            pass
                    elif cloth_info['class'] in predefined_shoes_cls:
                        if is_shoes_belong_to_people():
                            if people_descriptor[key][ind]['clothes']['shoes'] is None:
                                people_descriptor[key][ind]['clothes']['shoes'] = []
                            people_descriptor[key][ind]['clothes']['shoes'].append(cloth_info)
                        else:
                            pass
                    else:
                        pass

    return people_descriptor


def match_attribute_to_object(descriptor, descriptor_with_attr, iou_thresh):
    """
        Create by Haojun Jiang, 2021/10/17
        Function:
            1.Match the clothes to people, and remove the clothes that can not find a matching people
    """
    for key, value in descriptor.items():
        if key in descriptor_with_attr:
            for ind_wo_attr, item_wo_attr in enumerate(value):
                bbox_wo_attr = item_wo_attr['bbox'][:4]
                for item_with_attr in descriptor_with_attr[key]:
                    if item_with_attr['attr'] is not None:
                        bbox_with_attr = item_with_attr['bbox'][:4]
                        iou = IoU(bbox_wo_attr, bbox_with_attr)
                        if iou > iou_thresh:
                            descriptor[key][ind_wo_attr]['attr'] = item_with_attr['attr']

    return descriptor


def relative_spatial_location(descriptor, image_size):
    """
        Create by Haojun Jiang, 2021/10/17
        Function:
            1.Left/right/middle
            2.Top/bottom
            3.Front/back
        Args:
            1.descriptor
            2.image_size: (h, w)
    """
    ### left/right/middle
    ### top/bottom
    horizontal_thresh = 50
    vertical_thresh = 50
    for key, value in descriptor.items():
        if len(value) > 1:
            tmp_bbox_center = []
            for ind in range(len(value)):
                tmp_bbox_center.append(center_of_bbox(value[ind]['bbox']))
            tmp_bbox_center = np.array(tmp_bbox_center)
            xmin_ind, xmax_ind = np.argmin(tmp_bbox_center[:, 0]), np.argmax(tmp_bbox_center[:, 0])
            ymin_ind, ymax_ind = np.argmin(tmp_bbox_center[:, 1]), np.argmax(tmp_bbox_center[:, 1])
            if (tmp_bbox_center[xmax_ind, 0] - tmp_bbox_center[
                xmin_ind, 0]) > horizontal_thresh:  # avoid the little shift of bbox
                descriptor[key][xmin_ind]['spatial'].append('left')
                descriptor[key][xmax_ind]['spatial'].append('right')
                if len(value) == 3:
                    for ind in range(len(value)):
                        if ind not in [xmin_ind, xmax_ind]:
                            descriptor[key][ind]['spatial'].append('middle')
                            descriptor[key][ind]['spatial'].append('center')
                elif len(value) > 3:
                    for ind in range(len(value)):
                        if ind not in [xmin_ind, xmax_ind]:
                            if tmp_bbox_center[ind, 0] > image_size[1] * 3 / 4:
                                descriptor[key][ind]['spatial'].append('right')
                            elif tmp_bbox_center[ind, 0] < image_size[1] / 4:
                                descriptor[key][ind]['spatial'].append('left')
                            else:
                                descriptor[key][ind]['spatial'].append('middle')
                                descriptor[key][ind]['spatial'].append('center')
                else:
                    pass

            if (tmp_bbox_center[ymax_ind, 1] - tmp_bbox_center[
                ymin_ind, 1]) > vertical_thresh:  # avoid the little shift of bbox
                descriptor[key][ymax_ind]['spatial'].append('bottom')
                descriptor[key][ymin_ind]['spatial'].append('top')
                for ind in range(len(value)):
                    if ind not in [ymin_ind, ymax_ind]:
                        if tmp_bbox_center[ind, 1] > image_size[0] * 3 / 4:
                            descriptor[key][ind]['spatial'].append('bottom')
                        elif tmp_bbox_center[ind, 1] < image_size[1] / 4:
                            descriptor[key][ind]['spatial'].append('top')
                        else:
                            pass

    ### front/behind
    area_ratio_thresh_low = 0.4
    area_ratio_thresh_up = 0.8
    for key, value in descriptor.items():
        if len(value) > 1:
            tmp_bbox_area = []
            for ind in range(len(value)):
                tmp_bbox_area.append(area_of_bbox(value[ind]['bbox']))
            tmp_bbox_area = np.array(tmp_bbox_area)
            min_ind, max_ind = np.argmin(tmp_bbox_area), np.argmax(tmp_bbox_area)
            if tmp_bbox_area[min_ind] / tmp_bbox_area[max_ind] < area_ratio_thresh_low:
                descriptor[key][min_ind]['spatial'].append('behind')
                descriptor[key][max_ind]['spatial'].append('front')
                if len(value) > 3:
                    for ind in range(len(value)):
                        if ind not in [min_ind, max_ind]:
                            if tmp_bbox_area[ind] / tmp_bbox_area[max_ind] < area_ratio_thresh_low:
                                descriptor[key][ind]['spatial'].append('behind')
                            elif tmp_bbox_area[ind] / tmp_bbox_area[max_ind] > area_ratio_thresh_up:
                                descriptor[key][ind]['spatial'].append('front')
                            else:
                                pass

    return descriptor


def process_of_descriptor(people_descriptor, clothes_descriptor, things_descriptor, image_size):
    new_clothes_descriptor = []
    for key, value in clothes_descriptor.items():
        for item in value:
            new_clothes_descriptor.append(item)
    people_descriptor = match_clothes_to_people(people_descriptor, new_clothes_descriptor)
    descriptor = {}
    for key, value in people_descriptor.items():
        descriptor[key] = value
    for key, value in things_descriptor.items():
        descriptor[key] = value
    descriptor = relative_spatial_location(descriptor, image_size)
    return descriptor


def generate_description(descriptor, image_file, pseudo_train_samples, each_image_query=10):
    """
        Create by Haojun Jiang, 2021/10/18
        Function: generate pseudo training samples
    """
    all_candidate = []
    for object in descriptor:
        spatial_candidate = []
        for ind in range(len(object['spatial'])):
            spatial_candidate.append(object['spatial'][ind])

        ### Template 1: noun
        if object['attr'] is not None:
            description_string = '{} {}'.format(object['attr'], object['class'])
        else:
            description_string = '{}'.format(object['class'])
        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                   description_string, 'useless placeholder']
        all_candidate.append(tmp_pseudo_train_sample)

        ### Template 2/3: (noun rela) (rela noun)
        for ind in range(len(spatial_candidate)):
            if object['attr'] is not None:
                description_string = '{} {} {}'.format(object['attr'], object['class'], spatial_candidate[ind])
            else:
                description_string = '{} {}'.format(object['class'], spatial_candidate[ind])
            tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                       description_string, 'useless placeholder']
            all_candidate.append(tmp_pseudo_train_sample)

            if spatial_candidate[ind] in ['left', 'right']:
                if object['attr'] is not None:
                    description_string = '{} {} on the {}'.format(object['attr'], object['class'], spatial_candidate[ind])
                else:
                    description_string = '{} on the {}'.format(object['class'], spatial_candidate[ind])
                tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                           description_string, 'useless placeholder']
                all_candidate.append(tmp_pseudo_train_sample)

            if spatial_candidate[ind] in ['front', 'behind', 'middle', 'center']:
                if object['attr'] is not None:
                    description_string = '{} {} in the {}'.format(object['attr'], object['class'], spatial_candidate[ind])
                else:
                    description_string = '{} in the {}'.format(object['class'], spatial_candidate[ind])
                tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                           description_string, 'useless placeholder']
                all_candidate.append(tmp_pseudo_train_sample)

            if object['attr'] is not None:
                description_string = '{} {} {}'.format(spatial_candidate[ind], object['attr'], object['class'])
            else:
                description_string = '{} {}'.format(spatial_candidate[ind], object['class'])

            tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                       description_string, 'useless placeholder']
            all_candidate.append(tmp_pseudo_train_sample)

        if object['class'] in predefined_people_cls:
            clothes_candidate = []
            for key, value in object['clothes'].items():
                if value is not None:
                    for ind in range(len(value)):
                        if value[ind]['attr'] is not None:
                            clothes_candidate.append('{} {}'.format(value[ind]['attr'], value[ind]['class']))
                        else:
                            clothes_candidate.append(value[ind]['class'])

            ### Template 4/5: (noun attr) (attr noun)
            for ind in range(len(clothes_candidate)):
                # (noun attr)
                if object['attr'] is not None:
                    description_string = '{} {} {}'.format(object['attr'], object['class'], clothes_candidate[ind])
                else:
                    description_string = '{} {}'.format(object['class'], clothes_candidate[ind])
                tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                           description_string, 'useless placeholder']
                all_candidate.append(tmp_pseudo_train_sample)

                # (attr noun)
                if object['attr'] is not None:
                    description_string = '{} {} {}'.format(clothes_candidate[ind], object['attr'], object['class'])
                else:
                    description_string = '{} {}'.format(clothes_candidate[ind], object['class'])

                tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                           description_string, 'useless placeholder']
                all_candidate.append(tmp_pseudo_train_sample)

            ### three element: for people also have (noun attr rela)
            if object['attr'] is not None:
                candidate = ['{} {}'.format(object['attr'], object['class'])]
            else:
                candidate = [object['class']]
            for sind in range(len(spatial_candidate)):
                for cind in range(len(clothes_candidate)):
                    ### Template 6: {noun} {attr} {rela}
                    description_string = '{} {} {}'.format(candidate[0], clothes_candidate[cind],
                                                           spatial_candidate[sind])
                    tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                               description_string, 'useless placeholder']
                    all_candidate.append(tmp_pseudo_train_sample)

                    ### Template 7: {noun} {rela} {attr}
                    description_string = '{} {} {}'.format(candidate[0], spatial_candidate[sind],
                                                           clothes_candidate[cind])
                    tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                               description_string, 'useless placeholder']
                    all_candidate.append(tmp_pseudo_train_sample)

                    ### Template 8: {attr} {noun} {rela}
                    description_string = '{} {} {}'.format(clothes_candidate[cind], candidate[0],
                                                           spatial_candidate[sind])
                    tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                               description_string, 'useless placeholder']
                    all_candidate.append(tmp_pseudo_train_sample)

                    ### Template 9: {attr} {rela} {noun}
                    description_string = '{} {} {}'.format(clothes_candidate[cind], spatial_candidate[sind],
                                                           candidate[0])
                    tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                               description_string, 'useless placeholder']
                    all_candidate.append(tmp_pseudo_train_sample)

                    ### Template 10: {rela} {noun} {attr}
                    description_string = '{} {} {}'.format(spatial_candidate[sind], candidate[0],
                                                           clothes_candidate[cind])
                    tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                               description_string, 'useless placeholder']
                    all_candidate.append(tmp_pseudo_train_sample)

                    ### Template 11: {rela} {attr} {noun}
                    description_string = '{} {} {}'.format(spatial_candidate[sind], clothes_candidate[cind],
                                                           candidate[0])
                    tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                               description_string, 'useless placeholder']
                    all_candidate.append(tmp_pseudo_train_sample)

                    if spatial_candidate[sind] in ['left', 'right']:
                        ### Template 6: {noun} {attr} {rela}
                        description_string = '{} {} on the {}'.format(candidate[0], clothes_candidate[cind],
                                                                      spatial_candidate[sind])
                        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                                   description_string, 'useless placeholder']
                        all_candidate.append(tmp_pseudo_train_sample)

                        ### Template 7: {noun} {rela} {attr}
                        description_string = '{} on the {} {}'.format(candidate[0], spatial_candidate[sind],
                                                                      clothes_candidate[cind])
                        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                                   description_string, 'useless placeholder']
                        all_candidate.append(tmp_pseudo_train_sample)

                        ### Template 8: {attr} {noun} {rela}
                        description_string = '{} {} on the {}'.format(clothes_candidate[cind], candidate[0],
                                                                      spatial_candidate[sind])
                        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                                   description_string, 'useless placeholder']
                        all_candidate.append(tmp_pseudo_train_sample)

                    if spatial_candidate[sind] in ['front', 'behind', 'center', 'middle']:
                        ### Template 6: {noun} {attr} {rela}
                        description_string = '{} {} in the {}'.format(candidate[0], clothes_candidate[cind],
                                                                      spatial_candidate[sind])
                        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                                   description_string, 'useless placeholder']
                        all_candidate.append(tmp_pseudo_train_sample)

                        ### Template 7: {noun} {rela} {attr}
                        description_string = '{} in the {} {}'.format(candidate[0], spatial_candidate[sind],
                                                                      clothes_candidate[cind])
                        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                                   description_string, 'useless placeholder']
                        all_candidate.append(tmp_pseudo_train_sample)

                        ### Template 8: {attr} {noun} {rela}
                        description_string = '{} {} in the {}'.format(clothes_candidate[cind], candidate[0],
                                                                      spatial_candidate[sind])
                        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                                   description_string, 'useless placeholder']
                        all_candidate.append(tmp_pseudo_train_sample)

    if len(all_candidate) < each_image_query:
        return descriptor, pseudo_train_samples + all_candidate
    else:
        tmp_candidate = random.sample(all_candidate, each_image_query)
        return descriptor, pseudo_train_samples + tmp_candidate


def topn_conf_samples(descriptor, topn=3):
    """
        Create by Haojun Jiang, 2021/10/18
        Function: Select description to generate pseudo training samples
    """

    topn_people_samples = []
    samples_conf = []
    for key, value in descriptor.items():
        for ind in range(len(value)):
            if key in predefined_people_cls:
                topn_people_samples.append(value[ind])
                samples_conf.append(value[ind]['bbox'][-1])

    if len(topn_people_samples) > topn:
        delte_sample_ind = np.argsort(np.array(samples_conf))[:len(topn_people_samples) - topn]
        sorted_delte_sample_ind = sorted(delte_sample_ind, reverse=True)

        for ind in sorted_delte_sample_ind:
            topn_people_samples.pop(ind)

        return topn_people_samples

    else:
        topn -= len(topn_people_samples)
        topn_thing_samples = []
        samples_conf = []
        for key, value in descriptor.items():
            for ind in range(len(value)):
                if key not in predefined_people_cls:
                    topn_thing_samples.append(value[ind])
                    samples_conf.append(value[ind]['bbox'][-1])

        if len(topn_thing_samples) > topn:
            delte_sample_ind = np.argsort(np.array(samples_conf))[:len(topn_thing_samples) - topn]
            sorted_delte_sample_ind = sorted(delte_sample_ind, reverse=True)

            for ind in sorted_delte_sample_ind:
                topn_thing_samples.pop(ind)

        return topn_people_samples + topn_thing_samples


def bua_attr_detect(image_attr_detection_result, attr_conf_thresh):
    people_descriptor = []
    things_descriptor = []
    clothes_descriptor = []

    for object in image_attr_detection_result:
        cls = object[0]
        attr_conf = object[-1]
        if filter_detect_cls(cls):
            bbox = object[1][:4]
            conf = object[1][-1]
            if bbox[0] == 0:
                bbox[0] = 1
            if bbox[1] == 0:
                bbox[1] = 1

            if np.max(attr_conf) > attr_conf_thresh:
                attr_ind = np.argmax(attr_conf)
                detected_attr = bua_attributes[attr_ind + 1]
                if detected_attr in cls:
                    cls = cls.replace('{} '.format(detected_attr), '')
            else:
                detected_attr = None

            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                print('### Warning ###: Unvalid bounding box = {}, class = {}, conf = {}'.format(bbox, cls, conf))
            else:
                if is_cls_a_people(cls):
                    people_info = {'class': cls,
                                   'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                                            float(conf)],
                                   'clothes': {'hat': None, 'shirt': None, 'pants': None, 'shoes': None}, 'spatial': [],
                                   'attr': detected_attr}
                    people_descriptor.append(people_info)
                elif is_cls_a_cloth(cls):
                    cloth_info = {'class': cls,
                                  'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)],
                                  'spatial': [], 'attr': detected_attr}
                    clothes_descriptor.append(cloth_info)
                else:
                    thing_info = {'class': cls,
                                  'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)],
                                  'spatial': [], 'attr': detected_attr}
                    things_descriptor.append(thing_info)
        else:
            print('Ignore the class {}!'.format(cls))

    return people_descriptor, clothes_descriptor, things_descriptor


def object_detect(image_object_detection_result):
    people_descriptor = []
    things_descriptor = []
    clothes_descriptor = []

    for object in image_object_detection_result:
        cls = object[0]
        if filter_detect_cls(cls):
            bbox = object[1][:4]
            conf = object[1][-1]
            if bbox[0] == 0:
                bbox[0] = 1
            if bbox[1] == 0:
                bbox[1] = 1

            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                print('### Warning ###: Unvalid bounding box = {}, class = {}, conf = {}'.format(bbox, cls, conf))
            else:
                if is_cls_a_people(cls):
                    people_info = {'class': cls,
                                   'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                                            float(conf)],
                                   'clothes': {'hat': None, 'shirt': None, 'pants': None, 'shoes': None}, 'spatial': [],
                                   'attr': None}
                    people_descriptor.append(people_info)
                elif is_cls_a_cloth(cls):
                    cloth_info = {'class': cls,
                                  'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)],
                                  'attr': None, 'spatial': []}
                    clothes_descriptor.append(cloth_info)
                else:
                    thing_info = {'class': cls,
                                  'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf)],
                                  'attr': None, 'spatial': []}
                    things_descriptor.append(thing_info)
        else:
            print('Ignore the class {}!'.format(cls))

    return people_descriptor, clothes_descriptor, things_descriptor


if __name__ == '__main__':
    args = parse_args()

    if args.vg_dataset in ['unc', 'unc+', 'gref', 'gref_umd']:
        args.image_dir = os.path.join(args.vg_dataset_path, 'other/images/mscoco/images/train2014')
    elif args.vg_dataset == 'referit':
        args.image_dir = os.path.join(args.vg_dataset_path, 'referit/images')
    else:  # flickr
        args.image_dir = os.path.join(args.vg_dataset_path, 'Flickr30k/flickr30k-images/')

    args.out_path = '../data/pseudo_samples/{}'.format(args.vg_dataset)

    args.image_list_file = '../data/statistic/{}/{}_train_imagelist_split{}.txt'.format(
        args.vg_dataset, args.vg_dataset, args.split_ind)
    args.detection_file = '../data/detection_results/{}/r101_object_detection_results/{}_train_pseudo_split{}_detection_results.pth'.format(
        args.vg_dataset, args.vg_dataset, args.split_ind)
    args.attr_detection_file = '../data/detection_results/{}/r152_attr_detection_results/{}_train_pseudo_split{}_attr_detection_results.pth'.format(
        args.vg_dataset, args.vg_dataset, args.split_ind)

    train_image_list = open(args.image_list_file, 'r')
    train_image_files = train_image_list.readlines()
    off_the_shelf_object_detection_result = torch.load(args.detection_file)
    off_the_shelf_attr_detection_result = torch.load(args.attr_detection_file)
    pseudo_train_samples = []
    count = 0
    start_time = time.time()
    for image_ind, image_file in enumerate(train_image_files):
        if image_ind % 100 == 0:
            left_time = ((time.time() - start_time) * (len(train_image_files) - image_ind - 1) / (image_ind + 1)) / 3600
            print('Processing {}-th image, Left Time = {:.2f} hour ...'.format(image_ind, left_time))

        args.image_file = image_file[:-1]
        im_file = os.path.join(args.image_dir, args.image_file)
        im = np.array(imread(im_file))
        image_object_detection_result = off_the_shelf_object_detection_result[args.image_file]
        image_attr_detection_result = off_the_shelf_attr_detection_result[args.image_file]

        people_descriptor, clothes_descriptor, things_descriptor = object_detect(image_object_detection_result)
        people_descriptor = remove_tiny_bbox(people_descriptor, im.shape[:2])
        things_descriptor = remove_tiny_bbox(things_descriptor, im.shape[:2])
        people_descriptor = remove_overlap_bbox(people_descriptor)
        clothes_descriptor = remove_overlap_bbox(clothes_descriptor)
        things_descriptor = remove_overlap_bbox(things_descriptor)

        bua_people_descriptor, bua_clothes_descriptor, bua_things_descriptor = bua_attr_detect(
            image_attr_detection_result, attr_conf_thresh=args.attr_conf_thresh)
        bua_people_descriptor = remove_overlap_bbox(bua_people_descriptor)
        bua_clothes_descriptor = remove_overlap_bbox(bua_clothes_descriptor)
        bua_things_descriptor = remove_overlap_bbox(bua_things_descriptor)

        people_descriptor = match_attribute_to_object(people_descriptor, bua_people_descriptor,
                                                      iou_thresh=args.attr_iou_thresh)  # args.attr_iou_thresh default = 0.5
        clothes_descriptor = match_attribute_to_object(clothes_descriptor, bua_clothes_descriptor,
                                                       iou_thresh=args.attr_iou_thresh)  # args.attr_iou_thresh default = 0.5
        things_descriptor = match_attribute_to_object(things_descriptor, bua_things_descriptor,
                                                      iou_thresh=args.attr_iou_thresh)

        descriptor = process_of_descriptor(people_descriptor, clothes_descriptor, things_descriptor, im.shape[:2])  # image size: (h, w)
        descriptor = topn_conf_samples(descriptor, topn=args.topn)
        descriptor, pseudo_train_samples = generate_description(descriptor, args.image_file, pseudo_train_samples,
                                                                each_image_query=args.each_image_query)

    image_list_file = args.image_list_file
    output_path = os.path.join(args.out_path, 'top{}_query{}'.format(
        args.topn, args.each_image_query, args.attr_iou_thresh, args.attr_conf_thresh))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(pseudo_train_samples,
               os.path.join(output_path, '{}_train_pseudo_split{}.pth'.format(args.vg_dataset, args.split_ind)))
    print('Save file to {}'.format(
        os.path.join(output_path, '{}_train_pseudo_split{}.pth'.format(args.vg_dataset, args.split_ind))))
