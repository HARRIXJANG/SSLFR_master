from Model_utils import PointASISTransformer_All_attribute
import h5py
import numpy as np
from matplotlib import pyplot as plt
from DataProcessing.Utils import read_txt_and_normalize
from sklearn.cluster import MeanShift
import torch

face_colors = [
    (0.9700192609404378, 0.9055119492048388, 0.1323910958497898),
    (0.06660504960373947, 0.8303813089118807, 0.18731932715332889),
    (0.10215758587633339, 0.44758647359931925, 0.19743749570413038),
    (0.39618326204551335, 0.62480565418795, 0.49263998623974803),
    (0.9563194150570774, 0.6863431793453533, 0.40198773505084073),
    (0.7130311335430903, 0.5230673415079722, 0.360958551997956),
    (0.9546937583877466, 0.6021401628064251, 0.10398061899932864),
    (0.128418629621174, 0.38339751306229297, 0.19158928190370528),
    (0.9608394495112227, 0.8562415399879139, 0.35996379127307776),
    (0.8447461411950761, 0.6094638042385847, 0.6270924499592639),
    (0.608161974268185, 0.14829199916733193, 0.8045844806839375),
    (0.3911100021120745, 0.4512360980634469, 0.4243274963243149),
    (0.14587592017360218, 0.022838821343438398, 0.15571507918186522),
    (0.8096958445411236, 0.7164091463852411, 0.10006398944389583),
    (0.17637293645693264, 0.1958775455478048, 0.817706000786001),
    (0.44944192621774237, 0.738938573906961, 0.47097575885431253),
    (0.4988884139971932, 0.12540630349619342, 0.05117859638958533),
    (0.7141989735141261, 0.10619575782538193, 0.40160785621449757),
    (0.8907191760896118, 0.32853909664596714, 0.5617643232088937),
    (0.003188679730863675, 0.2513818008038544, 0.31507520557618907),
    (0.04338783996955187, 0.5109066219752398, 0.01751921372339693),
    (0.08918523237871268, 0.09105427694981261, 0.2694775316636171),
    (0.6080768096407021, 0.34579812513326547, 0.8826508065977654),
    (0.4926405898863041, 0.9728342822717221, 0.9958939931665864),
    (0.65, 0.65, 0.7)
]

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
    Face_points, _, _, _, _, data_face_id = load_one_part(training_file, id)
    Face_points=np.expand_dims(Face_points[:, :-2], axis=0)
    Face_points_tensor = torch.tensor(Face_points, dtype=torch.float32)
    Face_id_tensor = torch.tensor(data_face_id,  dtype=torch.float32)
    return Face_points_tensor, Face_id_tensor

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

def model_forward(points, model, para_path):
    model.eval()
    model.load_model_from_ckpt(para_path)
    pts_tensor = torch.tensor(points)
    with torch.no_grad():
        ret_ss, ret_isf = model(pts_tensor)
    return ret_ss, ret_isf

def draw_types(types_f, gt_coors, title):
    x = gt_coors[:, 0]
    y = gt_coors[:, 1]
    z = gt_coors[:, 2]

    feature_type = types_f

    colors = ["violet", "#3283FF", "purple", "#494F80"]

    types = dict()

    for i in range(x.shape[0]):
        if i == 0:
            types[feature_type[i]] = np.array([[x[i], y[i], z[i]]])
        else:
            if feature_type[i] in types.keys():
                temp_v = types[feature_type[i]]
                types[feature_type[i]] = np.concatenate((temp_v, np.array([[x[i], y[i], z[i]]])), axis=0)
            else:
                types[feature_type[i]] = np.array([[x[i], y[i], z[i]]])
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
        v_arr = np.array(v)
        x_temp = v_arr[:, 0]
        y_temp = v_arr[:, 1]
        z_temp = v_arr[:, 2]
        # 绘制点云
        ax.scatter(x_temp, y_temp, z_temp, c=color, marker='.')

    ax.set_axis_off()
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    plt.title(title)
    plt.show()

def draw_semantics(types_f, gt_coors, face_ids, title):
    data_face_id = np.array(face_ids, dtype=int)
    all_face_ids = np.unique(data_face_id)
    face_dict_pred = dict()
    colors = ["violet", "#3283FF", "purple", "#494F80"]

    for id in all_face_ids:
        if id not in face_dict_pred.keys():
            face_dict_pred[id] = []

    for i in range(data_face_id.shape[0]):
        face_dict_pred[data_face_id[i]].append(types_f[i])

    def vote(face_dict_pred):
        for k, v in face_dict_pred.items():
            v_m = max(set(v), key=v.count)
            face_dict_pred[k] = v_m
        return face_dict_pred

    f_d_p = vote(face_dict_pred)

    x = gt_coors[:, 0]
    y = gt_coors[:, 1]
    z = gt_coors[:, 2]

    types = dict()
    for i, ins_type in enumerate(f_d_p.values()):
        types[ins_type] = []

    for i in range(x.shape[0]):
        face_id = int(face_ids[i])
        ins_type = f_d_p[face_id]
        types[ins_type].append(np.array([[x[i], y[i], z[i]]]))
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
        v_arr = np.squeeze(np.array(v))

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

def draw_ins(ins, gt_coors, face_ids, title):
    data_face_id = np.array(face_ids, dtype=int)
    all_face_ids = np.unique(data_face_id)
    face_dict_pred = dict()

    for id in all_face_ids:
        if id not in face_dict_pred.keys():
            face_dict_pred[id] = []

    for i in range(data_face_id.shape[0]):
        face_dict_pred[data_face_id[i]].append(ins[i])

    def vote(face_dict_pred):
        for k,v in face_dict_pred.items():
            v_m = max(set(v), key=v.count)
            face_dict_pred[k] = v_m
        return face_dict_pred

    f_d_p = vote(face_dict_pred)

    x = gt_coors[:, 0]
    y = gt_coors[:, 1]
    z = gt_coors[:, 2]

    ins_labels = dict()
    for i, ins_type in enumerate(f_d_p.values()):
        ins_labels[ins_type] = []

    for i in range(x.shape[0]):
        face_id = int(face_ids[i])
        ins_type = f_d_p[face_id]
        ins_labels[ins_type].append(np.array([[x[i], y[i], z[i]]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (k, v) in enumerate(ins_labels.items()):
        if i < len(face_colors):
            color = face_colors[i]
        else:
            color = face_colors[i - len(face_colors)]
        v_arr = np.squeeze(np.array(v))

        x_temp = v_arr[:, 0]
        y_temp = v_arr[:, 1]
        z_temp = v_arr[:, 2]
        # 绘制点云
        ax.scatter(x_temp, y_temp, z_temp, c=color, marker='.')

    ax.set_axis_off()
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    plt.title(title)
    plt.show()

def draw_semantics_after_ins(types_f, ins, gt_coors, title):
    all_ins_ids = np.unique(ins)
    face_dict_pred = dict()
    colors = ["violet", "#3283FF", "purple", "#494F80"]

    for id in all_ins_ids:
        if id not in face_dict_pred.keys():
            face_dict_pred[id] = []

    for i in range(ins.shape[0]):
        face_dict_pred[ins[i]].append(types_f[i])

    def vote(face_dict_pred):
        for k, v in face_dict_pred.items():
            v_m = max(set(v), key=v.count)
            face_dict_pred[k] = v_m
        return face_dict_pred

    f_d_p = vote(face_dict_pred)

    x = gt_coors[:, 0]
    y = gt_coors[:, 1]
    z = gt_coors[:, 2]

    types = dict()
    for i, sem_type in enumerate(f_d_p.values()):
        types[sem_type] = []

    for i in range(x.shape[0]):
        ins_id = int(ins[i])
        sem_type = f_d_p[ins_id]
        types[sem_type].append(np.array([[x[i], y[i], z[i]]]))
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
        v_arr = np.squeeze(np.array(v))

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

def Predict_for_one_part(points, face_ids, model, para_path):
    ret_ss, ret_isf = model_forward(points, model, para_path)
    ret_isf = ret_isf.transpose(2, 1)
    ret_ss_numpy = ret_ss.to('cpu').detach().numpy()
    ret_isf_numpy = ret_isf.to('cpu').detach().numpy()

    ret_ss_ids = np.argmax(ret_ss_numpy, axis=-1).squeeze()

    # draw_semantics(ret_ss_ids, points[0,:, :3], face_ids,"semantic_seg_orig")
    p_l, _, flag = mean_shift(ret_isf_numpy[0])
    print(np.max(p_l))
    if flag == 1:
        draw_ins(p_l, points[0, :, :3], face_ids, "instance_seg")
        draw_semantics_after_ins(ret_ss_ids, p_l, points[0, :, :3], "semantic_seg")



def predict_for_txt(txt_path, para_path):
    part, face_id = read_txt_and_normalize(txt_path, 1024)
    part_tensor = torch.tensor(part[np.newaxis, :], dtype=torch.float32)
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

    Predict_for_one_part(part_tensor, face_id, base_model, para_path)

def predict_for_h5(h5_path, part_id, para_path):
    part,face_id = read_one_part(h5_path, part_id)
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
    Predict_for_one_part(part, face_id, base_model, para_path)

if __name__ == "__main__":
    # Replace with your path
    txt_data_path = r"F:\SSLFR_master\demo_parts\pts\Part622.txt"
    para_path = r"F:\SSLFR_master\fine_tuning_experiment_path\Fine_tune_1.0.pth"
    valid_data_path = r"F:\SSLFR_master\data\validation.h5"

    # predict_for_txt(txt_data_path, para_path)
    predict_for_h5(valid_data_path, 878, para_path)