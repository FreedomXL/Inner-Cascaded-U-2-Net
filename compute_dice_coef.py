
import seg_metrics.seg_metrics as sg


def compute_hausdorff_distance(groundtruth_path, pred_path):

    labels = [1]

    csv_file = 'metrics.csv'

    metrics = sg.write_metrics(labels, groundtruth_path, pred_path, csv_file, metrics=['dice', 'hd95'])
    return metrics['dice'], metrics['hd95']


if __name__ == '__main__':

    dataset = 'LiTS'  # change dataset to 'BraTS' when testing with BraTS 2013 dataset

    if dataset == 'LiTS':
        for idx in range(115, 130):
            # the tumor areas are all labelled as liver for binary segmentation
            groundtruth_path = 'LiTS/true_01/' + str(idx) + '.nii.gz'
            pred_path = 'LiTS/pred/pred_threshold_' + str(idx) + '.nii.gz'
            dice, hd95 = compute_hausdorff_distance(groundtruth_path, pred_path)
            print(str(idx) + ":")
            print('dice:', dice)
            print('hd95:', hd95)
            print()
    elif dataset == 'BraTS':
        # here's my choice of test set from random selection
        path = ['HG-0005', 'HG-0011', 'HG-0015', 'HG-0027', 'LG-0008', 'LG-0013']
        for idx in range(6):
            groundtruth_path = 'BraTS2013/resized_labels/' + path[idx] + '.nii.gz'
            pred_path = 'BraTS2013/pred/pred_threshold_' + path[idx] + '.nii.gz'
            dice, hd95 = compute_hausdorff_distance(groundtruth_path, pred_path)
            print(path[idx] + ':')
            print('dice:', dice)
            print('hd95:', hd95)
            print()
