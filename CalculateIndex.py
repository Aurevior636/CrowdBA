import os
import numpy as np
import pandas as pd

def compute_iou(true_boxes: np.ndarray, predict_box: np.ndarray) -> np.ndarray:
    x_min = np.maximum(true_boxes[:, 0], predict_box[0])
    y_min = np.maximum(true_boxes[:, 1], predict_box[1])
    x_max = np.minimum(true_boxes[:, 2], predict_box[2])
    y_max = np.minimum(true_boxes[:, 3], predict_box[3])

    true_boxes_areas = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    predict_box_area = (predict_box[2] - predict_box[0]) * (predict_box[3] - predict_box[1])

    intersection = np.maximum(0.0, x_max - x_min) * np.maximum(0.0, y_max - y_min)
    ious = intersection / (true_boxes_areas + predict_box_area - intersection)

    return ious

def voc_eval_per_img(detections, ground_truths, ovthresh=0.5):
    img_ids = ground_truths['img_id'].unique()
    results = []
    iou_list = []  # 用于存储每张图片的平均 IoU
    precision_list = []  # 用于存储每张图片的精准率
    recall_list = []  # 用于存储每张图片的召回率
    f1_list = []  # 用于存储每张图片的 F1-score

    for img_id in img_ids:
        img_ground_truths = ground_truths[ground_truths['img_id'] == img_id]
        img_detections = detections[detections['img_id'] == img_id]

        # 跳过空的检测和真实框
        if img_ground_truths.empty or img_detections.empty:
            continue

        n_gt = len(img_ground_truths)
        n_det = len(img_detections)

        # 初始化匹配标记
        gt_detected = np.zeros(n_gt)
        tp = 0
        fp = 0

        selected_ious = []  # 用于存储匹配的 IoU

        for d in range(n_det):
            bb = img_detections.iloc[d][['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float)
            ious = compute_iou(img_ground_truths[['xmin', 'ymin', 'xmax', 'ymax']].values, bb)
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= ovthresh and gt_detected[max_iou_idx] == 0:
                tp += 1
                gt_detected[max_iou_idx] = 1
                selected_ious.append(max_iou)
            else:
                fp += 1

        fn = n_gt - np.sum(gt_detected)
        # 计算 Precision, Recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        mean_iou = np.mean(selected_ious) if selected_ious else 0.0
        iou_list.append(mean_iou)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)

        results.append((img_id, recall, precision, mean_iou, f1_score))

    # 计算平均值时确保非空列表
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    mean_f1 = np.mean(f1_list) if f1_list else 0.0
    mean_precision = np.mean(precision_list) if precision_list else 0.0
    mean_recall = np.mean(recall_list) if recall_list else 0.0

    return results, mean_iou, mean_f1, mean_precision, mean_recall

if __name__ == '__main__':
    output_file = "C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/output/wbf/wbf-output-iouthr0.7.csv"
    ground_truths_file = "C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/ground-truth/ground-truth.csv"

    detections = pd.read_csv(output_file)
    ground_truths = pd.read_csv(ground_truths_file)

    results, mean_iou, mean_f1, mean_precision, mean_recall = voc_eval_per_img(detections, ground_truths)

    # 将结果保存为 CSV 文件
    results_df = pd.DataFrame(results, columns=['Image ID', 'Recall', 'Precision', 'Mean IoU', 'F1-score'])
    results_df.to_csv('evaluation_results.csv', index=False)

    # 输出汇总指标
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean F1-score: {mean_f1:.4f}")
    print(f"Mean Precision(误检): {mean_precision:.4f}")
    print(f"Mean Recall(漏检): {mean_recall:.4f}")
