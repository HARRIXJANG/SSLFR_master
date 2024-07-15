from sklearn.cluster import MeanShift
from DataProcessing import Utils
from Model_utils import PointASISTransformer_All_attribute, PointASISTransFormer_only_for_ins
import torch
import numpy as np
from torch.utils import data
from Fine_tuning import read_data, read_AAG_data
from scipy.optimize import linear_sum_assignment
import time

def model_forward(points, model, para_path, batch_size, device):
    model.load_model_from_ckpt(para_path)
    model.to(device)
    pts_tenor = torch.tensor(points)
    dataloader = data.DataLoader(dataset=pts_tenor,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 )
    count = 0
    for idx, pts in enumerate(dataloader):
        with torch.no_grad():
            pts = pts.to(device)
            ret_ss, ret_isf = model(pts)
            if count == 0:
                all_ret_ss = ret_ss
                all_ret_isf = ret_isf
            else:
                all_ret_ss = torch.concat((all_ret_ss, ret_ss), dim=0)
                all_ret_isf = torch.concat((all_ret_isf, ret_isf), dim=0)
            count += 1

    return all_ret_ss, all_ret_isf

def model_forward_only_ins(points, model, para_path, batch_size, device):
    model.load_model_from_ckpt(para_path)
    model.to(device)
    pts_tenor = torch.tensor(points)
    dataloader = data.DataLoader(dataset=pts_tenor,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 )
    count = 0
    for idx, pts in enumerate(dataloader):
        with torch.no_grad():
            pts = pts.to(device)
            ret_isf = model(pts)
            if count == 0:
                all_ret_isf = ret_isf
            else:
                all_ret_isf = torch.concat((all_ret_isf, ret_isf), dim=0)
            count += 1

    return all_ret_isf

def mean_shift(ret_isf):
    bandwidth = 1.282
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    try:
        ms.fit(ret_isf)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        return labels, cluster_centers, 1
    except:
        return np.nan, np.nan, 0

def get_instance_set(instance_label):
    L = len(instance_label)
    instance_label_array = np.array(instance_label)
    label_indices = np.arange(L)
    label_mask = np.ones(L, dtype=bool)
    instance_idxs = []
    while True:
        instance_idx = []
        L = instance_label_array.shape[0]
        real_i = label_indices[0]
        instance_idx.append(real_i)
        for j in range(1, L):
            if instance_label_array[0] == instance_label_array[j]:
                real_j = label_indices[j]
                instance_idx.append(real_j)
                label_mask[j] = False
        label_mask[0] = False
        instance_label_array = instance_label_array[label_mask]
        label_indices = label_indices[label_mask]
        label_mask = label_mask[label_mask]
        instance_idxs.append(set(instance_idx))
        if label_mask.shape[0] == 0:
            break
    return instance_idxs

def get_instance_matrix(instance_label):
    L = len(instance_label)
    instance_label_array = np.array(instance_label)
    label_indices = np.arange(L)
    label_mask = np.ones(L, dtype=bool)

    matrix = np.eye(L, dtype=int)
    while True:
        instance_idx = []
        L = instance_label_array.shape[0]
        real_i = label_indices[0]
        instance_idx.append(real_i)
        for j in range(1, L):
            if instance_label_array[0] == instance_label_array[j]:
                real_j = label_indices[j]
                instance_idx.append(real_j)
                label_mask[j] = False
        label_mask[0] = False
        instance_label_array = instance_label_array[label_mask]
        label_indices = label_indices[label_mask]
        label_mask = label_mask[label_mask]
        for k in range(len(instance_idx)):
            k_idx = instance_idx[k]
            instance_idx_array = np.array(instance_idx)
            mask_instance_idx = np.ones_like(instance_idx_array, dtype=bool)
            mask_instance_idx[k] = False
            temp_instance_idx = instance_idx_array[mask_instance_idx]
            matrix[k_idx][temp_instance_idx] = 1

        if label_mask.shape[0] == 0:
            break
    return matrix

def Iou_for_points(p_l, g_t):
    g_t = np.array(g_t, dtype=int)
    p_l_indices = extract_indices(p_l.tolist())
    g_t_indices = extract_indices(g_t.tolist())
    average_iou, row_ind_preds, col_ind_gts, all_correct_pred = calculate_multiple_iou(p_l_indices, g_t_indices)
    return average_iou

def Acc_for_faces(p_l, g_t, data_face_id, ret_ss_id, true_label_id):
    data_face_id = np.array(data_face_id, dtype=int)
    g_t = np.array(g_t, dtype=int)
    all_face_ids = np.unique(data_face_id)
    face_dict_pred = dict()
    face_dict_gt = dict()
    face_p_semantic = dict()
    face_t_semantic = dict()

    for id in all_face_ids:
        if id not in face_dict_pred.keys():
            face_dict_pred[id] = []
            face_dict_gt[id] = None
            face_p_semantic[id] = []
            face_t_semantic[id] = None

    for i in range(data_face_id.shape[0]):
        face_dict_pred[data_face_id[i]].append(p_l[i])
        if face_dict_gt[data_face_id[i]] == None:
            face_dict_gt[data_face_id[i]] = g_t[i]

        face_p_semantic[data_face_id[i]].append(ret_ss_id[i])
        if face_t_semantic[data_face_id[i]] == None:
            face_t_semantic[data_face_id[i]] = true_label_id[i]

    def vote(face_dict_pred):
        for k, v in face_dict_pred.items():
            v_m = max(set(v), key=v.count)
            face_dict_pred[k] = v_m
        return face_dict_pred

    f_d_p = vote(face_dict_pred)
    f_p_res = list(f_d_p.values())
    g_t_res = list(face_dict_gt.values())
    p_l_set = get_instance_set(f_p_res)
    g_t_set = get_instance_set(g_t_res)

    average_iou, row_ind_preds, col_ind_gts, all_correct_pred = calculate_multiple_iou(p_l_set, g_t_set)

    f_d_p_sem = vote(face_p_semantic)
    f_p_sem = list(f_d_p_sem.values())
    g_t_sem = list(face_t_semantic.values())
    correct_1 = np.equal(f_p_sem, g_t_sem)
    acc_sem = np.sum(correct_1) / (len(g_t_sem))

    return average_iou, acc_sem

def Iou_for_faces(p_l, g_t, data_face_id):
    data_face_id = np.array(data_face_id, dtype=int)
    g_t = np.array(g_t, dtype=int)
    face_dict_pred = dict()
    face_dict_gt = dict()
    for i in range(data_face_id.shape[0]):
        face_dict_pred[data_face_id[i]].append(p_l[i])
        if face_dict_gt[data_face_id[i]] == None:
            face_dict_gt[data_face_id[i]] = g_t[i]

    def vote(face_dict_pred):
        for k, v in face_dict_pred.items():
            v_m = max(set(v), key=v.count)
            face_dict_pred[k] = v_m
        return face_dict_pred

    f_d_p = vote(face_dict_pred)
    f_p_res = list(f_d_p.values())
    g_t_res = list(face_dict_gt.values())
    p_l_set = get_instance_set(f_p_res)
    g_t_set = get_instance_set(g_t_res)

    average_iou, row_ind_preds, col_ind_gts, all_correct_pred = calculate_multiple_iou(p_l_set, g_t_set)
    return average_iou


def extract_indices(arr):
    index_dict = {}
    for index, value in enumerate(arr):
        if value in index_dict:
            index_dict[value].append(index)
        else:
            index_dict[value] = [index]
    all_indices = []
    for v in index_dict.values():
        all_indices.append(set(v))
    return all_indices


def calculate_iou(set1, set2):
    """
    Calculate the IoU between two sets of IDs.

    Args:
    set1 (set): A set of IDs.
    set2 (set): A set of IDs.

    Returns:
    float: IoU value.
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union


def calculate_multiple_iou(pred_sets, gt_sets):
    """
    Calculate the IoU for multiple sets of IDs between prediction and ground truth.

    Args:
    pred_sets (list of sets): List of sets of predicted IDs.
    gt_sets (list of sets): List of sets of ground truth IDs.

    Returns:
    float: Average IoU.
    """
    num_preds = len(pred_sets)
    num_gts = len(gt_sets)

    # Create IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    for i, pred in enumerate(pred_sets):
        for j, gt in enumerate(gt_sets):
            iou_matrix[i, j] = calculate_iou(pred, gt)

    # Apply Hungarian Algorithm for optimal matching
    row_ind_preds, col_ind_gts = linear_sum_assignment(-iou_matrix)  # maximization problem

    # Calculate average IoU for matched pairs
    all_correct_pred = []
    total_iou = 0.0
    for i, j in zip(row_ind_preds, col_ind_gts):
        total_iou += iou_matrix[i, j]
        if iou_matrix[i, j] == 1.:
            all_correct_pred.append(i)

    average_iou = total_iou / len(row_ind_preds)
    return average_iou, row_ind_preds, col_ind_gts, all_correct_pred


def Acc_for_all_points(points, data_face_id, g_t, true_label, model, para_path, batch_size, device='cuda'):
    '''
    Returns:
        acc_sem: semantic segment accuracy for point clouds
        all_iou_points/count: instance segment IoU for point clouds
    '''
    ret_ss, ret_isf = model_forward(points, model, para_path, batch_size, device)
    ret_isf = ret_isf.transpose(2, 1)
    g_t_numpy = g_t.to('cpu').detach().numpy()
    data_face_id = data_face_id.to('cpu').detach().numpy()
    true_label_numpy = true_label.to('cpu').detach().numpy()
    ret_ss_numpy = ret_ss.to('cpu').detach().numpy()
    ret_isf_numpy = ret_isf.to('cpu').detach().numpy()

    true_label_ids = np.argmax(true_label_numpy, axis=-1).squeeze()
    ret_ss_ids = np.argmax(ret_ss_numpy, axis=-1).squeeze()
    correct_1 = np.equal(true_label_ids, ret_ss_ids)
    acc_sem = np.sum(correct_1) / (ret_ss_ids.shape[0]*ret_ss_ids.shape[1])
    print("acc_points: " + str(acc_sem))

    all_iou_points = 0
    count = 0

    for i in range(points.shape[0]):
        p_l, _, flag = mean_shift(ret_isf_numpy[i])
        if flag == 1:
            acc_iou_points = Iou_for_points(p_l, g_t_numpy[i], data_face_id[i], ret_ss_ids[i], true_label_ids[i])
            all_iou_points += acc_iou_points
            count+=1
            if (i % 50 == 0)&(i != 0):
                print("########################################")
                print("iou_points:" + str(all_iou_points / count))
                print(i)

    return acc_sem, all_iou_points/count

def Acc_for_all_faces(points, data_face_id, g_t, true_label, model, para_path, batch_size, device='cuda'):
    '''
    Returns:
        all_iou_faces / count: instance segment IoU for faces
        all_acc_faces / count: semantic segment accuracy for faces
    '''
    ret_ss, ret_isf = model_forward(points, model, para_path, batch_size, device)
    ret_isf = ret_isf.transpose(2, 1)
    g_t_numpy = g_t.to('cpu').detach().numpy()
    data_face_id = data_face_id.to('cpu').detach().numpy()
    true_label_numpy = true_label.to('cpu').detach().numpy()
    ret_ss_numpy = ret_ss.to('cpu').detach().numpy()
    ret_isf_numpy = ret_isf.to('cpu').detach().numpy()

    true_label_ids = np.argmax(true_label_numpy, axis=-1).squeeze()
    ret_ss_ids = np.argmax(ret_ss_numpy, axis=-1).squeeze()

    all_iou_faces = 0
    all_acc_faces = 0
    count = 0

    for i in range(points.shape[0]):
        p_l, _, flag = mean_shift(ret_isf_numpy[i])
        if flag == 1:
            iou_faces, acc_faces = Acc_for_faces(p_l, g_t_numpy[i], data_face_id[i], ret_ss_ids[i], true_label_ids[i])
            all_iou_faces += iou_faces
            all_acc_faces += acc_faces
            count+=1
            if (i % 50 == 0)&(i != 0):
                print("########################################")
                print("iou_faces:" + str(all_iou_faces / count))
                print("acc_faces:" + str(all_acc_faces / count))
                print(i)

    return all_acc_faces / count, all_iou_faces / count

def Acc_only_ins_for_faces(points, data_face_id, g_t, model, para_path, batch_size, device='cuda'):
    '''
    Returns:
        all_acc_iou_points/count: instance segment IoU for faces
    '''
    ret_isf = model_forward_only_ins(points, model, para_path, batch_size, device)
    ret_isf = ret_isf.transpose(2, 1)
    g_t_numpy = g_t.to('cpu').detach().numpy()
    data_face_id = data_face_id.to('cpu').detach().numpy()
    ret_isf_numpy = ret_isf.to('cpu').detach().numpy()
    all_iou_face = 0
    count = 0
    for i in range(points.shape[0]):
        p_l, _, flag = mean_shift(ret_isf_numpy[i])
        if flag == 1:
            iou_face = Iou_for_faces(p_l, g_t_numpy[i], data_face_id[i])
            all_iou_face += iou_face
            count += 1
            if (i % 50 == 0) & (i != 0):
                print("########################################")
                print("iou_face:" + str(all_iou_face / count))
                print(i)

    return all_iou_face/count

if __name__ == '__main__':
#########################################################Test_for_ASIN_data#############################################
    para_path = r"F:\ZH\AllSelfLearningCode\SSLFR_master\fine_tuning_experiment_path\Fine_tune_0.5.pth"
    valid_h5_path = r"F:\ZH\AllSelfLearningCode\SSLFR_master\data\validation.h5"
    base_model = PointASISTransformer_All_attribute(
        trans_dim=384,
        encoder_depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        encoder_dim=384,
        group_size=32,
        num_group=64,
        points_attribute=10,
    )

    s_t = time.time()
    All_points_tensor, data_point_face, data_point_edge, data_part_id, data_class_label, data_instance_label, data_face_id = read_data(valid_h5_path, False)
    
    '''
    If you want to calculate the metrics for faces, Please change the function 'Acc_for_all_points' to 'Acc_for_all_faces'
    '''
    
    acc_sem_points, iou_ins_points = Acc_for_all_points(data_point_face,
                                                        data_face_id,
                                                        data_instance_label,
                                                        data_class_label,
                                                        base_model,
                                                        para_path,
                                                        64,
                                                        device='cuda'
                                                        )
    e_t = time.time()
    print(e_t - s_t)
#######################################################Test_for_MFInstSeg_data##########################################
    # para_path = r"F:\ZH\AllSelfLearningCode\SSLFR_master\fine_tuning_experiment_path\Fine_tune_0.5_MFInstSeg.pth"
    #
    # valid_h5_path = r"F:\ZH\AllSelfLearningCode\SSLFR_master\data\valid_MFInstSeg_pt.h5"
    #
    # Face_points_tensor_v, points_face_ids, Instance_label_tensor_v, Class_label_tensor_v = read_AAG_data(valid_h5_path)
    # Class_label_tensor_v_one_hot = torch.nn.functional.one_hot((Class_label_tensor_v).to(int))
    # _, _, class_num = Class_label_tensor_v_one_hot.shape
    # base_model = PointASISTransFormer_only_for_ins(
    #     trans_dim=384,
    #     encoder_depth=12,
    #     drop_path_rate=0.1,
    #     num_heads=6,
    #     encoder_dim=384,
    #     group_size=32,
    #     num_group=64,
    #     points_attribute=10,
    #     class_num=class_num,
    #     ins_out_dim=10,
    # )
    # s_t = time.time()
    # iou_ins_faces = Acc_only_ins_for_faces(Face_points_tensor_v,
    #                                        points_face_ids,
    #                                        Instance_label_tensor_v,
    #                                        base_model,
    #                                        para_path,
    #                                        64
    #                                        )
    # e_t = time.time()
    # print(e_t-s_t)
