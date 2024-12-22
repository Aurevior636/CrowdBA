# nms.py
import numpy as np
import pandas as pd
from numba import jit

iou_threshold = 0.95
input_file = 'C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/simulate/simulated_boxes0.5.csv'
output_file = f'C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/output/nms/nms-output-iouthr{iou_threshold}.csv'

def load_boxes_from_csv(filepath):
    data = pd.read_csv(filepath)
    img_ids = data['img_id'].unique()
    img_boxes = {img_id: data[data['img_id'] == img_id][['xmin', 'ymin', 'xmax', 'ymax']].values for img_id in img_ids}
    img_scores = {img_id: data[data['img_id'] == img_id]['score'].values for img_id in img_ids} if 'score' in data.columns else {img_id: np.ones(len(data[data['img_id'] == img_id])) for img_id in img_ids}
    img_labels = {img_id: data[data['img_id'] == img_id]['label'].values for img_id in img_ids}
    return img_boxes, img_scores, img_labels, img_ids

def prepare_boxes(boxes, scores, labels):
    result_boxes = boxes.copy()
    result_boxes[:, 0] = np.min(result_boxes[:, [0, 2]], axis=1)
    result_boxes[:, 2] = np.max(result_boxes[:, [0, 2]], axis=1)
    result_boxes[:, 1] = np.min(result_boxes[:, [1, 3]], axis=1)
    result_boxes[:, 3] = np.max(result_boxes[:, [1, 3]], axis=1)

    area = (result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1])
    cond = (area == 0)
    if cond.sum() > 0:
        result_boxes = result_boxes[area > 0]
        scores = scores[area > 0]
        labels = labels[area > 0]
    return result_boxes, scores, labels

@jit(nopython=True)
def nms_float_fast(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nms_method(boxes, scores, labels, weights=None):
    if weights is not None:
        if len(boxes) != len(weights):
            return [], [], []
        weights = np.array(weights)
        for i in range(len(weights)):
            scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()

    filtered_boxes, filtered_scores, filtered_labels = [], [], []
    for i in range(len(boxes)):
        if len(boxes[i]) == 0:
            continue
        filtered_boxes.append(boxes[i])
        filtered_scores.append(scores[i])
        filtered_labels.append(labels[i])

    boxes = np.concatenate(filtered_boxes)
    scores = np.concatenate(filtered_scores)
    labels = np.concatenate(filtered_labels)

    boxes, scores, labels = prepare_boxes(boxes, scores, labels)

    unique_labels = np.unique(labels)
    final_boxes, final_scores, final_labels = [], [], []
    for l in unique_labels:
        condition = (labels == l)
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        labels_by_label = np.array([l] * len(boxes_by_label))

        keep = nms_float_fast(boxes_by_label, scores_by_label, thresh=iou_threshold)
        if len(keep) > 0:
            final_boxes.append(boxes_by_label[keep])
            final_scores.append(scores_by_label[keep])
            final_labels.append(labels_by_label[keep])

    if len(final_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)
    return final_boxes, final_scores, final_labels

def nms(boxes, scores, labels, weights=None):
    return nms_method(boxes, scores, labels, weights=weights)

def save_boxes_to_csv(filepath, img_id, final_boxes, final_labels):
    rows = []
    for box, label in zip(final_boxes, final_labels):
        rows.append({'img_id': img_id, 'label': label, 'xmin': box[0], 'xmax': box[2], 'ymin': box[1], 'ymax': box[3]})
    df = pd.DataFrame(rows, columns=['img_id', 'label', 'xmin', 'xmax', 'ymin', 'ymax'])
    df.to_csv(filepath, mode='a', header=False, index=False)

if __name__ == '__main__':
    input_file = input_file
    output_file = output_file
    img_boxes, img_scores, img_labels, img_ids = load_boxes_from_csv(input_file)
    df = pd.DataFrame(columns=['img_id', 'label', 'xmin', 'xmax', 'ymin', 'ymax'])
    df.to_csv(output_file, index=False)
    for img_id in img_ids:
        boxes = img_boxes[img_id]
        scores = img_scores[img_id]
        labels = img_labels[img_id]
        final_boxes, final_scores, final_labels = nms([boxes], [scores], [labels])
        save_boxes_to_csv(output_file, img_id, final_boxes, final_labels)
    print(f"Results saved to {output_file}")