# coding: utf-8

import warnings
import numpy as np
import pandas as pd
from numba import jit
import numpy.random as npr

iou_threshold = 0.95
input_file = 'C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/simulate/simulated_boxes0.5.csv'
output_file = f'C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/output/nmw/nmw-output-iouthr{iou_threshold}.csv'

@jit(nopython=True)
def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    new_boxes = dict()
    for t in range(len(boxes)):
        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]),
                                                                                                       len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]),
                                                                                                       len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            b = [int(label), float(score) * weights[t], x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes):
    box = np.zeros(6, dtype=np.float32)
    best_box = boxes[0]
    conf = 0
    for b in boxes:
        iou = bb_intersection_over_union(b[2:], best_box[2:])
        weight = b[1] * iou
        box[2:] += (weight * b[2:])
        conf += weight
    box[0] = best_box[0]
    box[1] = best_box[1]
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def non_maximum_weighted(boxes_list, scores_list, labels_list, weights=None, iou_thr=iou_threshold, skip_box_thr=0.0):
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights),
                                                                                                     len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights) / max(weights)

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        main_boxes = []

        while len(boxes) > 0:
            # 随机选择一个基准框
            random_index = npr.choice(len(boxes))
            current_box = boxes[random_index]

            index, best_iou = find_matching_box(main_boxes, current_box, iou_thr)
            if index != -1:
                new_boxes[index].append(current_box.copy())
            else:
                new_boxes.append([current_box.copy()])
                main_boxes.append(current_box.copy())

            boxes = np.delete(boxes, random_index, axis=0)

        weighted_boxes = []
        for j in range(0, len(new_boxes)):
            box = get_weighted_box(new_boxes[j])
            weighted_boxes.append(box.copy())

        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels


def main(input_file, output_file):
    data = pd.read_csv(input_file)
    img_ids = data['img_id'].unique()

    output_data = []

    for img_id in img_ids:
        img_data = data[data['img_id'] == img_id]
        boxes = img_data[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        # Check if 'score' column exists, if not, set scores to 1
        if 'score' in img_data.columns:
            scores = img_data['score'].values.tolist()
        else:
            scores = [1.0] * len(boxes)

        labels = img_data['label'].values.tolist()

        boxes, scores, labels = non_maximum_weighted([boxes], [scores], [labels])

        for box, label in zip(boxes, labels):
            output_data.append([img_id, int(label), box[0], box[2], box[1], box[3]])

    output_df = pd.DataFrame(output_data, columns=['img_id', 'label', 'xmin', 'xmax', 'ymin', 'ymax'])
    output_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = input_file
    output_file = output_file
    main(input_file, output_file)