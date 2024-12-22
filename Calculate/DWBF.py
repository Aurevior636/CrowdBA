import numpy as np
import pandas as pd

iou_threshold = 0.95
input_file = 'C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/simulate/simulated_boxes0.5.csv'
output_file = f'C:/Users/87946/Desktop/Weighted-Boxes-Fusion-master/cocoData/output/wbf/wbf-output-iouthr{iou_threshold}.csv'

# Function to calculate IoU between an array of boxes and a new box
def bb_iou_array(boxes, new_box):
    xA = np.maximum(boxes[:, 0], new_box[0])
    yA = np.maximum(boxes[:, 2], new_box[2])
    xB = np.minimum(boxes[:, 1], new_box[1])
    yB = np.minimum(boxes[:, 3], new_box[3])

    interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

    boxAArea = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    boxBArea = (new_box[1] - new_box[0]) * (new_box[3] - new_box[2])

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

# Function to calculate IoU between two boxes
def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[1], boxB[1])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2])
    boxBArea = (boxB[1] - boxB[0]) * (boxB[3] - boxB[2])

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

# Function to assign clusters based on IoU
def assign_clusters(data, iou_threshold=iou_threshold):
    clusters = []
    cluster_indices = []

    for img_id in data['img_id'].unique():
        img_data = data[data['img_id'] == img_id]
        boxes = img_data[['label', 'xmin', 'xmax', 'ymin', 'ymax']].values
        clusters_for_img = []

        for box in boxes:
            matched = False
            for cluster in clusters_for_img:
                ious = bb_iou_array(np.array([c[2:] for c in cluster]), box[1:])
                if np.any(ious > iou_threshold):
                    cluster.append((img_id, *box))
                    matched = True
                    break
            if not matched:
                clusters_for_img.append([(img_id, *box)])

        for i, cluster in enumerate(clusters_for_img):
            clusters.extend(cluster)
            cluster_indices.extend([i] * len(cluster))

    return clusters, cluster_indices

# Function to calculate IoU weights with exponential decay
def calculate_iou_weights(data, exponent=2):
    data['score'] = 0.0  # Initialize score column

    for img_id in data['img_id'].unique():
        img_data = data[data['img_id'] == img_id]
        for cluster in img_data['cluster'].unique():
            cluster_data = img_data[img_data['cluster'] == cluster]
            if len(cluster_data) == 1:
                data.loc[cluster_data.index, 'score'] = 1.0
            else:
                iou_sums = np.zeros(len(cluster_data))
                for i in range(len(cluster_data)):
                    boxA = cluster_data.iloc[i][['xmin', 'xmax', 'ymin', 'ymax']].values
                    for j in range(len(cluster_data)):
                        if i != j:
                            boxB = cluster_data.iloc[j][['xmin', 'xmax', 'ymin', 'ymax']].values
                            iou = bb_iou(boxA, boxB)
                            iou_sums[i] += iou
                # Apply exponential decay to the IoU sums
                exp_iou_sums = np.exp(-exponent * iou_sums)
                scores = exp_iou_sums / np.sum(exp_iou_sums)
                data.loc[cluster_data.index, 'score'] = [round(score, 5) for score in scores]

    return data

# Function to compute weighted bounding box
def get_weighted_box(boxes, use_max_weight=False):
    if use_max_weight:
        # Select the box with the maximum score
        max_weight_box = max(boxes, key=lambda b: b[0])
        return max_weight_box[1:]

    # Compute weighted average of bounding box coordinates
    box = np.zeros(4, dtype=np.float32)
    total_weight = 0
    for b in boxes:
        weight = b[0]  # Use score as weight
        box += (weight * b[1:])
        total_weight += weight
    box /= total_weight  # Normalize
    return box

# Function to compute weighted label
def get_weighted_label(group):
    label_scores = {}
    for _, row in group.iterrows():
        label = row['label']
        weight = row['score']  # Use score as weight
        if label not in label_scores:
            label_scores[label] = 0
        label_scores[label] += weight  # Sum weights for each label

    # Select the label with the highest total weight
    selected_label = max(label_scores, key=label_scores.get)
    return selected_label

# Main execution
if __name__ == "__main__":
    # Read the input CSV file
    input_csv_path = input_file
    output_csv_path = output_file

    print(f"Reading input file: {input_csv_path}")
    data = pd.read_csv(input_csv_path)
    print("Input file read successfully.")

    # Step 1: Assign clusters based on IoU
    print("Assigning clusters based on IoU...")
    clusters, cluster_indices = assign_clusters(data)
    print("Clusters assigned.")

    # Add the cluster indices to the DataFrame
    clustered_data = pd.DataFrame(clusters, columns=['img_id', 'label', 'xmin', 'xmax', 'ymin', 'ymax'])
    clustered_data['cluster'] = cluster_indices

    # Step 2: Calculate IoU weights
    print("Calculating IoU weights...")
    clustered_data = calculate_iou_weights(clustered_data)
    print("IoU weights calculated.")

    # Step 3: Group by img_id and cluster, and perform weighted box fusion
    print("Performing weighted box fusion...")
    grouped = clustered_data.groupby(['img_id', 'cluster'])

    # Store results
    results = []

    for name, group in grouped:
        img_id = name[0]
        cluster = name[1]
        if len(group) < 2:
            # Skip clusters with less than 2 boxes
            continue

        boxes = group[['score', 'xmin', 'xmax', 'ymin', 'ymax']].values
        label = get_weighted_label(group)  # Compute weighted label

        # Perform weighted box fusion
        weighted_box = get_weighted_box(boxes)
        xmin, xmax, ymin, ymax = weighted_box

        results.append([img_id, label, xmin, xmax, ymin, ymax, cluster])

    # Convert to DataFrame and save results
    result_df = pd.DataFrame(results, columns=['img_id', 'label', 'xmin', 'xmax', 'ymin', 'ymax', 'cluster'])
    result_df.to_csv(output_csv_path, index=False)
    print(f"Weighted box fusion completed and saved to {output_csv_path}")
