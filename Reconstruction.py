from Model_utils import Double_Decoder_MAE_All_attribute
import h5py
import torch
import numpy as np
from matplotlib import pyplot as plt
from DataProcessing.Utils import read_txt_and_normalize

def load_one_part(h5_filename, id):
    with h5py.File(h5_filename, 'r') as file:
        data_point_edge = file['point_edge'][id]
        data_point_face = file['point_face'][id]
        data_part_id = file['part_id'][id]
        data_class_label = file['class_label'][id]
        data_instance_label = file['instance_label'][id]
        data_face_id = file['point_face_id'][id]

    return  data_point_face, data_point_edge, data_part_id, data_class_label, data_instance_label, data_face_id

def read_one_part(training_file, id):
    Face_points, _, _, _, _, _ = load_one_part(training_file, id)
    Face_points=np.expand_dims(Face_points[:, :-2], axis=0)
    Face_points_tensor = torch.tensor(Face_points, dtype=torch.float32)
    return Face_points_tensor

def load_one_part_for_AAGNET(h5_filename, id):
    with h5py.File(h5_filename, 'r') as file:
        points_coors = file['point_coors'][id]
        points_face_ids = file['points_face_ids'][id]
    return points_coors, points_face_ids

def read_one_part_from_AAGNET(training_file, id):
    Face_points, _, = load_one_part_for_AAGNET(training_file, id)
    Face_points_tensor = torch.tensor(Face_points, dtype=torch.float32)
    return Face_points_tensor

def draw_coordiantes(coors, title, color='b'):
    x = coors[:, 0]
    y = coors[:, 1]
    z = coors[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color, marker='.')
    ax.set_axis_off()
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    plt.title(title)
    plt.show()

def Denoise_coors(gt_coors, all_coors, threshold):
    cloud1_expanded = np.expand_dims(gt_coors, axis=1)
    cloud2_expanded = np.expand_dims(all_coors, axis=0)
    distances = np.linalg.norm(cloud1_expanded - cloud2_expanded, axis=-1)

    min_distances_2 = np.min(distances, axis=0)
    index = np.squeeze(np.argwhere(min_distances_2>threshold))
    print(index.shape[0])
    new_coors = np.delete(all_coors, index, axis=0)
    return new_coors

def distance(x, y):
    d = np.sqrt(np.sum((x - y) * (x - y)))
    return d

def if_exist(x, all_x):
     flag = 0
     for x_i in all_x:
         if distance(x, x_i)<0.001:
             flag = 1
             break
     return flag

def Unique_coors(gt_coors):
    unique_c = []
    unique_ids = []
    for i in range(gt_coors.shape[0]):
        if i == 0:
            unique_c.append(gt_coors[i])
            unique_ids.append(i)
        else:
            if if_exist(gt_coors[i], unique_c)==0:
                unique_c.append(gt_coors[i])
                unique_ids.append(i)
    # draw_coordiantes(np.array(unique_c))
    return np.array(unique_c), np.array(unique_ids)

def Denoise_normals(gt_coors, nors):
    all_gt_coors, all_gt_ids = Unique_coors(gt_coors)
    nors_den = nors[all_gt_ids]
    all_nors = dict()
    gt_coors_den = np.zeros_like(all_gt_coors)
    for i in range(all_gt_coors.shape[0]):
        all_nors[i] = 0.
        gt_coors_den[i]=all_gt_coors[i]
    for i in range(gt_coors.shape[0]):
        for j in range(all_gt_coors.shape[0]):
            if distance(gt_coors[i], all_gt_coors[j])<0.001:
                all_nors[j] = (nors[i]+all_nors[j])/2
    return gt_coors_den, nors_den, np.array(list(all_nors.values()))

def draw_normals(nors, gt_coors, title, coor_color, nor_color):
    x = gt_coors[:, 0]
    y = gt_coors[:, 1]
    z = gt_coors[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=coor_color, marker='.')
    ax.set_axis_off()
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    scale_factor = 0.1
    for i in range(len(gt_coors)):
        start = gt_coors[i]
        ax.quiver(start[0], start[1], start[2], scale_factor*nors[i, 0], scale_factor*nors[i, 1], scale_factor*nors[i, 2], color=nor_color)
    plt.title(title)
    plt.show()

def draw_types(types_f, gt_coors, title):
    x = gt_coors[:, 0]
    y = gt_coors[:, 1]
    z = gt_coors[:, 2]
    face_type = types_f

    colors = ["violet", "#3283FF", "purple", "#494F80"]

    types = dict()

    for i in range(x.shape[0]):
        if i == 0:
            types[face_type[i]] = np.array([[x[i], y[i], z[i]]])
        else:
            if face_type[i] in types.keys():
                temp_v = types[face_type[i]]
                types[face_type[i]] = np.concatenate((temp_v, np.array([[x[i], y[i], z[i]]])), axis=0)
            else:
                types[face_type[i]] = np.array([[x[i], y[i], z[i]]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (k, v) in enumerate(types.items()):
        if k == 0:
            color = colors[0]
        elif k == 1:
            color = colors[1]
        elif k == 2:
            color = colors[2]
        elif k == 3:
            color = colors[3]
        else:
            color = color[0]
        v_arr = np.array(v)
        x_temp = v_arr[:, 0]
        y_temp = v_arr[:, 1]
        z_temp = v_arr[:, 2]
        ax.scatter(x_temp, y_temp, z_temp, c=color, marker='.')

    ax.set_axis_off()
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    plt.title(title)
    plt.show()

def construction_point_cloud(pt, model):
    all_coors, all_attrs, coors_vis, attr_vis, all_groudtruth_coors = model(pt, vis=True)
    pt = pt.squeeze().detach().numpy()
    origin_coors = pt[:, :3]
    origin_nors = pt[:, 3:6]
    origin_types = pt[:, 6:]
    draw_coordiantes(origin_coors, "origin_coors",color="#3283FF")#3283FF

    all_coors = all_coors.squeeze().detach().numpy()
    all_attrs = all_attrs.squeeze().detach().numpy()
    coors_vis = coors_vis.squeeze().detach().numpy()
    attr_vis = attr_vis.squeeze().detach().numpy()
    all_groudtruth_coors = all_groudtruth_coors.squeeze().detach().numpy()

    gt_all_coors = all_attrs[:, :3]
    gt_vis_coors = attr_vis[:, :3]

    all_coors_de = Denoise_coors(all_coors, gt_all_coors, 0.1)
    draw_coordiantes(all_coors_de, "output_coors", color="#002CB0")
    draw_coordiantes(coors_vis, "mask_coors", color="#494F80")

    nors_all = all_attrs[:, 3:6]
    nors_vis = attr_vis[:, 3:6]

    gt_coors_den, nors_den, _ = Denoise_normals(gt_all_coors, nors_all)
    draw_normals(origin_nors, origin_coors, "origin_nors","#3283FF", "#3283FF")
    draw_normals(nors_den, gt_coors_den, "output_nors", "#002CB0", "#002CB0")
    draw_normals(nors_vis, gt_vis_coors, "mask_nors", "#494F80", "#494F80")
    origin_types = np.argmax(origin_types, axis=-1)
    types_all = np.argmax(all_attrs[:, 6:], axis=-1)
    types_vis =  np.argmax(attr_vis[:, 6:], axis=-1)
    draw_types(origin_types, origin_coors, "origin_types")
    draw_types(types_all, all_groudtruth_coors, "output_types")
    draw_types(types_vis, gt_vis_coors, "mask_types")

def construction_point_cloud_coor(pt, model):
    all_coors, all_vis, all_ground_truth, _ = model(pt, vis=True)
    pt = pt.squeeze().detach().numpy()
    origin_coors = pt[:, :3]
    draw_coordiantes(origin_coors, "origin_coors",color="#3283FF")

    all_coors = all_coors.squeeze().detach().numpy()
    coors_vis = all_vis.squeeze().detach().numpy()
    all_ground_truth = all_ground_truth.squeeze().detach().numpy()

    all_coors_de = Denoise_coors(all_coors, all_ground_truth, 0.1)
    draw_coordiantes(all_coors_de, "output_coors", color="#002CB0")
    draw_coordiantes(coors_vis, "mask_coors", color="#494F80")


def main_for_h5(h5_path, part_id, para_path):
    '''
    Reconstruct from h5
    Args:
        h5_path: h5 file of parts
        part_id: the id of the part to be reconstructed
        para_path: model parameters
    Returns:

    '''
    part = read_one_part(h5_path, part_id)
    base_model = Double_Decoder_MAE_All_attribute(
        mask_ratio=0.6,
        trans_dim=384,
        encoder_depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        encoder_dim=384,
        mask_type='rand',
        group_size=32,
        num_group=64,
        decoder_depth=4,
        decoder_num_heads=6,
        points_attribute=10
        )
    ckpt = torch.load(para_path)
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict=True)
    construction_point_cloud(part, base_model)

def main_for_txt(txt_path, para_path):
    '''
    Reconstruct from txt
    Args:
        txt_path: txt of a part
        para_path: model parameters
    '''
    part, _ = read_txt_and_normalize(txt_path, 1024)
    part_tensor = torch.tensor(part[np.newaxis, :], dtype=torch.float32)
    base_model = Double_Decoder_MAE_All_attribute(mask_ratio=0.9,
                           trans_dim=384,
                           encoder_depth=12,
                           drop_path_rate=0.1,
                           num_heads=6,
                           encoder_dim=384,
                           mask_type='rand',
                           group_size=32,
                           num_group=64,
                           decoder_depth=4,
                           decoder_num_heads=6,
                           points_attribute=10)
    ckpt = torch.load(para_path)
    base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict=True)
    construction_point_cloud(part_tensor, base_model)


if __name__ == "__main__":
    # Replace with your path
    train_data_path = r"F:\SSLFR_master\data\train.h5"
    valid_data_path = r"F:\SSLFR_master\data\validation.h5"

    txt_data_path = r"F:\SSLFR_master\demo_parts\pts\Part486.txt"
    para_path = r"F:\SSLFR_master\experiment_path\Pretrain_para.pth"

    main_for_h5(valid_data_path, 10, para_path)
    #main_for_txt(txt_data_path, para_path)
