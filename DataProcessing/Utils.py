import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import h5py
import random
import math

feature_type = 4

def read_data_from_txt(file_path):
    with open(file_path) as f:
        face_points = []
        edge_points = []
        lines = f.readlines()
        def string_to_float(s):
            return float(s)
        for line in lines:
            line = list(map(string_to_float, line.strip('\n').strip(' ').split(' ')))
            if line[0] == 0:
                face_points.append(line)
            else:
                edge_points.append(line)

        face_points_array = np.array(face_points)
        edge_points_array = np.array(edge_points)

        return face_points_array, edge_points_array

def read_sim_data_from_txt(file_path):
    with open(file_path) as f:
        face_ids = []
        all_face_ids = []
        lines = f.readlines()
        def string_to_int(s):
            return int(s)
        for line in lines:
            line = list(map(string_to_int, line.strip('\n').strip(' ').split(' ')))
            face_ids.append(line)
            for l in line:
                all_face_ids.append(l)
        return face_ids, all_face_ids

def save_h5(h5_file_name, part_id, point_face, point_edge, face_id=[], class_label=[], sim_matrix_label=[], sim_ins_label=[]):
    with h5py.File(h5_file_name, 'w') as h5_fout:
        data_dtype = 'float'
        id_dtype = 'int'

        h5_fout.create_dataset(
            'point_face', data=point_face,
            compression='gzip',
            dtype=data_dtype)
        h5_fout.create_dataset(
            'point_edge', data=point_edge,
            compression='gzip',
            dtype=data_dtype)
        h5_fout.create_dataset(
            'part_id', data=part_id,
            compression='gzip',
            dtype=id_dtype)
        if len(face_id) != 0:
            h5_fout.create_dataset(
                'point_face_id', data=face_id,
                compression='gzip',
                dtype=id_dtype
            )
        if len(class_label) != 0:
            label_dtype = 'float'
            h5_fout.create_dataset(
                'class_label', data=class_label,
                compression='gzip',
                dtype=label_dtype)
        if len(sim_matrix_label) != 0:
            label_dtype = 'float'
            h5_fout.create_dataset(
                'instance_label', data=sim_matrix_label,
                compression='gzip',
                dtype=label_dtype)
        if len(sim_ins_label) != 0:
            label_dtype = 'float'
            h5_fout.create_dataset(
                'instance_label', data=sim_ins_label,
                compression='gzip',
                dtype=label_dtype)

        h5_fout.close()

def load_h5(h5_filename):
    with h5py.File(h5_filename, 'r') as file:
        data_part_id = file['part_id'][:]
        data_point_edge = file['point_edge'][:]
        data_point_face = file['point_face'][:]

    return  data_part_id, data_point_edge, data_point_face

def load_h5_for_pretraining(h5_filename):
    with h5py.File(h5_filename, 'r') as file:
        data_point_edge = file['point_edge'][:]
        data_point_face = file['point_face'][:]

    return  data_point_face, data_point_edge

def load_h5_for_ASIN_data(h5_filename):
    with h5py.File(h5_filename, 'r') as file:
        data_point_edge = file['point_edge'][:]
        data_point_face = file['point_face'][:]
        data_part_id = file['part_id'][:]
        data_class_label = file['class_label'][:]
        data_instance_label = file['instance_label'][:]
        data_face_id = file['point_face_id'][:]

    return data_point_face, data_point_edge, data_part_id, data_class_label, data_instance_label, data_face_id

def load_h5_for_AAG_attr_data(h5_filename):
    with h5py.File(h5_filename, 'r') as file:
        points_coors = file['point_attrs'][:]
        points_face_ids = file['points_face_ids'][:]
        points_ins_labels = file['points_ins_labels'][:]
        points_sem_labels = file['points_sem_labels'][:]
    return points_coors, points_face_ids, points_ins_labels, points_sem_labels


def load_h5_write_h5(h5_paths, h5_file):
    _, all_files = get_all_files_in_folder(h5_paths)
    k = 0
    for file in all_files:
        data_part_id, data_point_edge, data_point_face = load_h5(file)
        if k == 0:
            data_part_ids = data_part_id
            data_point_edges = data_point_edge
            data_point_faces = data_point_face
        else:
            data_part_ids = np.concatenate((data_part_ids, data_part_id), axis=0)
            data_point_edges = np.concatenate((data_point_edges, data_point_edge), axis=0)
            data_point_faces = np.concatenate((data_point_faces, data_point_face), axis=0)
        k += 1
    save_h5(h5_file, data_part_ids, data_point_faces, data_point_edges)

def load_h5_write_h5_2(h5_paths, h5_file, num_face_points):
    _, all_files = get_all_files_in_folder(h5_paths)
    k = 0
    all_num_list = range(num_face_points)
    random.seed(1997)
    all_num_d = np.array(random.sample(all_num_list, num_face_points))
    for file in all_files:
        data_part_id, data_point_edge, data_point_face = load_h5(file)
        if k == 0:
            data_part_ids = data_part_id
            data_point_edges = data_point_edge
            data_point_faces = data_point_face[:, all_num_d, :]
        else:
            data_part_ids = np.concatenate((data_part_ids, data_part_id), axis=0)
            data_point_edges = np.concatenate((data_point_edges, data_point_edge), axis=0)
            data_point_faces = np.concatenate((data_point_faces, data_point_face[:, all_num_d, :]), axis=0)
        k += 1
    save_h5(h5_file, data_part_ids, data_point_faces, data_point_edges)

# data rotate
# points_nor = None, if there is no points normals, i.e. points in edges.
def data_rotate(points_coo, points_nor, angle, axis):
    angle_rad = np.radians(angle)

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cos_theta, -sin_theta],
                                    [0, sin_theta, cos_theta]])
    elif axis == 'y':
        rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                                    [0, 1, 0],
                                    [-sin_theta, 0, cos_theta]])
    elif axis == 'z':
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                    [sin_theta, cos_theta, 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid rotation axis. Use 'x', 'y', or 'z'.")

    rotation_point_coo = np.dot(points_coo, rotation_matrix.T) #transpose
    if len(points_nor) != 0:
        rotation_point_nor = np.dot(points_nor, rotation_matrix.T)
    else:
        rotation_point_nor = []

    return rotation_point_coo, rotation_point_nor

def data_argument_1(sampled_face_points, sampled_edge_points, get_matrix=0): #Get the instance matrix label at the same time.
    rotate_face_points = [sampled_face_points]
    rotate_edge_points = [sampled_edge_points]
    x_axis = [90, 180, 270]
    y_axis = [90, 270]
    for a_x in x_axis:
        rotation_point_coo_f, rotation_point_nor_f = data_rotate(sampled_face_points[:, 1:4], sampled_face_points[:, 4:7],
                                                             a_x, 'x')
        points_f = np.concatenate((sampled_face_points[:, 0][:, np.newaxis], rotation_point_coo_f, rotation_point_nor_f, sampled_face_points[:, 7:]), axis=-1)
        rotate_face_points.append(points_f)

        rotation_point_coo_e, _ = data_rotate(sampled_edge_points[:, 1:4], [], a_x, 'x')
        points_e = np.concatenate((sampled_edge_points[:, 0][:, np.newaxis], rotation_point_coo_e, sampled_edge_points[:, 4:]), axis=-1)
        rotate_edge_points.append(points_e)

    for a_y in y_axis:
        rotation_point_coo_f, rotation_point_nor_f = data_rotate(sampled_face_points[:, 1:4], sampled_face_points[:, 4:7],
                                                             a_y, 'y')
        points_f = np.concatenate((sampled_face_points[:, 0][:, np.newaxis], rotation_point_coo_f, rotation_point_nor_f, sampled_face_points[:, 7:]), axis=-1)
        rotate_face_points.append(points_f)

        rotation_point_coo_e, _ = data_rotate(sampled_edge_points[:, 1:4], [], a_y, 'y')
        points_e = np.concatenate((sampled_edge_points[:, 0][:, np.newaxis], rotation_point_coo_e, sampled_edge_points[:, 4:]), axis=-1)
        rotate_edge_points.append(points_e)

    point_instance_label = sampled_face_points[:, 9]

    if get_matrix==0:
        point_instance_matrix = []
    else:
        point_instance_matrix = get_instance_matrix(point_instance_label)

    return rotate_face_points, rotate_edge_points, point_instance_matrix, point_instance_label


def farthest_point_sample_cpu(points_coo, num_points_to_select):
    num_points = points_coo.shape[0]
    np.random.seed(1998)
    selected_indices = [np.random.randint(0, num_points)]  # Start with a random point index

    while len(selected_indices) < num_points_to_select:
        distances = np.min(np.linalg.norm(points_coo[selected_indices][:, np.newaxis] - points_coo, axis=2), axis=0)
        farthest_index = np.argmax(distances)
        selected_indices.append(farthest_index)

    return points_coo[selected_indices], selected_indices

def random_point_sampling(points_coo, num_points_to_select):
    num_points = points_coo.shape[0]
    np.random.seed(1998)
    if num_points>=num_points_to_select:
        selected_indices = np.random.choice(num_points, size=num_points_to_select, replace=False)
    else:
        selected_indices = np.random.choice(num_points, size=num_points_to_select, replace=True)
    return points_coo[selected_indices], selected_indices

def coordinate_normalization_1(point_cloud_coor):
    max_range = 1
    min_range = 0
    # Find the min and max coordinates along each axis
    min_coords = np.min(point_cloud_coor, axis=0)
    max_coords = np.max(point_cloud_coor, axis=0)

    # Calculate the scaling factors for each axis
    scale_factors = (max_range - min_range) / (max_coords - min_coords)
    if ((max_coords[0]-min_coords[0])==0)|((max_coords[1]-min_coords[1])==0)|((max_coords[2]-min_coords[2])==0):
        print('error')

    # Apply the scaling to the point cloud
    normalized_point_cloud = (point_cloud_coor - min_coords) * scale_factors + min_range

    return normalized_point_cloud

def coordinate_normalization_2(point_cloud_coor):
    max_range = 1
    min_range = 0
    # Find the min and max coordinates
    min_coords = np.min(point_cloud_coor)
    max_coords = np.max(point_cloud_coor)

    # Calculate the scaling factors for each axis
    scale_factors = (max_range - min_range) / (max_coords - min_coords)

    # Apply the scaling to the point cloud
    normalized_point_cloud = (point_cloud_coor - min_coords) * scale_factors + min_range

    return normalized_point_cloud

def one_hot(nums, all_num):
    one_hot_num = np.zeros((nums.shape[0], all_num))
    for i in range(nums.shape[0]):
        one_hot_num[i][int(nums[i])]= 1

    return one_hot_num

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

def read_txt_and_normalize(txt_path, face_points_num):
    face_points, _ = read_data_from_txt(txt_path)
    point_face_coor = face_points[:, 1:4]
    sampled_face_points = np.zeros((face_points_num, face_points.shape[1]))
    if face_points_num <= face_points.shape[0]:
        _, face_points_selected_indices = random_point_sampling(point_face_coor, face_points_num)
    else:
        _, face_points_selected_indices = np.concatenate(np.arange(0, point_face_coor.shape[0], dtype=int),
                                                         np.arange(0, face_points_num - point_face_coor.shape[0],
                                                                   dtype=int))
    i = 0
    for index in face_points_selected_indices:
        sampled_face_points[i] = face_points[index]
        i += 1

    num_f, _ = sampled_face_points.shape
    point_face_stand = np.zeros((num_f, 2))
    for i in range(num_f):
        point_face_idx = sampled_face_points[i, 0]
        point_face_stand[i][int(point_face_idx)] = 1.0

    point_face_coor = sampled_face_points[:, 1:4]
    point_face_nor = sampled_face_points[:, 4:7]
    point_face_id = sampled_face_points[:, 7]
    point_face_type = sampled_face_points[:, 8][:, np.newaxis]
    nor_point_face_coor = coordinate_normalization_1(point_face_coor)
    nor_point_face_nor = point_face_nor
    nor_point_face_type = one_hot(point_face_type, 4)
    normal_face_point = np.concatenate((nor_point_face_coor, nor_point_face_nor, nor_point_face_type), axis=-1)
    return normal_face_point, point_face_id


def standardlize_point_cloud(face_points, edge_points):
    num_f, _= face_points.shape
    point_face_stand = np.zeros((num_f, 2))
    for i in range(num_f):
        point_face_idx = face_points[i, 0]
        point_face_stand[i][int(point_face_idx)] = 1.0

    point_face_coor = face_points[:, 1:4]
    point_face_nor = face_points[:, 4:7]
    point_face_id = face_points[:, 7]
    point_face_type = face_points[:, 8][:, np.newaxis]

    point_face_label =  face_points[:, 10]
    point_face_label_stand = np.zeros((num_f, feature_type))
    for i in range(num_f):
        idx = int(np.log10(point_face_label[i]))
        point_face_label_stand[i, feature_type-1-idx] = 1

    num_e, _=edge_points.shape
    point_edge_stand = np.zeros((num_e, 2))
    for i in range(num_e):
        point_edge_idx = edge_points[i, 0]
        point_edge_stand[i][int(point_edge_idx)] = 1.0

    point_edge_coor = edge_points[:, 1:4]
    point_edge_type = edge_points[:, 4][:, np.newaxis]
    point_edge_id = edge_points[:, 5][:, np.newaxis]

    point_face_id_labels = list(set(point_face_id))
    encoder = OneHotEncoder(sparse=False)
    nor_point_face_id_nopad = encoder.fit_transform(point_face_id.reshape(-1, 1))
    nor_point_face_id_pad = np.zeros((point_face_stand.shape[0], 65-len(point_face_id_labels)))


    nor_point_face_coor = coordinate_normalization_1(point_face_coor)
    nor_point_face_nor = point_face_nor
    nor_point_face_type = one_hot(point_face_type, 4)
    nor_point_face_id = np.hstack((nor_point_face_id_nopad, nor_point_face_id_pad))

    nor_point_edge_coor = coordinate_normalization_1(point_edge_coor)
    nor_point_edge_type = one_hot(point_edge_type-1, 2)

    edge_pad_nor = np.zeros((point_edge_stand.shape[0], 3))
    edge_pad_type = np.zeros((point_edge_stand.shape[0], 2))
    edge_pad_id = np.zeros((point_edge_stand.shape[0], 65))
    edge_pad_id[-1] = 1

    final_face_point = np.hstack((nor_point_face_coor, nor_point_face_nor, nor_point_face_type, point_face_stand))
    final_edge_point = np.hstack((nor_point_edge_coor, edge_pad_nor, nor_point_edge_type, edge_pad_type, point_edge_stand))

    return final_face_point, final_edge_point, point_face_id, point_face_label_stand

def sample_from_face_and_edge_points(face_points, edge_points, face_points_num, edge_points_num, get_matrix=0):
    point_face_coor = face_points[:, 1:4]
    point_edge_coor = edge_points[:, 1:4]
    sampled_face_points = np.zeros((face_points_num, face_points.shape[1]))
    sampled_edge_points = np.zeros((edge_points_num, edge_points.shape[1]))
    if face_points_num <= face_points.shape[0]:
        _, face_points_selected_indices = random_point_sampling(point_face_coor, face_points_num)
    else:
        _, face_points_selected_indices = np.concatenate(np.arange(0, point_face_coor.shape[0], dtype=int),
                                                         np.arange(0, face_points_num - point_face_coor.shape[0],
                                                                   dtype=int))


    if edge_points_num <= edge_points.shape[0]:
        _, edge_points_selected_indices = random_point_sampling(point_edge_coor, edge_points_num)
    else:
        _, edge_points_selected_indices = np.concatenate(np.arange(0, point_edge_coor.shape[0]),
                                                         np.arange(0, edge_points_num - point_edge_coor.shape[0],
                                                                   dtype=int))

    i = 0
    for index in face_points_selected_indices:
        sampled_face_points[i] = face_points[index]
        i += 1
    j = 0
    for index in edge_points_selected_indices:
        sampled_edge_points[j] = edge_points[index]
        j += 1

    all_sampled_face_points, all_sampled_edge_points, point_instance_matrix,point_instance_label = data_argument_1(sampled_face_points, sampled_edge_points, get_matrix)

    return all_sampled_face_points, all_sampled_edge_points, point_instance_matrix, point_instance_label

def sample_from_face_and_edge_points_face_based(face_points, edge_points, face_points_num, edge_points_num, get_matrix=0):
    point_face_coor = face_points[:, 1:4]
    point_edge_coor = edge_points[:, 1:4]
    face_ids = face_points[:, 7]
    all_face_ids = np.unique(face_ids)

    each_face_points_num = math.ceil(face_points_num/len(all_face_ids))
    i = 0
    for t_id in all_face_ids:
        one_point_face_coor_temp = []
        k = 0
        for p_id in face_ids:
            if t_id==int(p_id):
                one_point_face_coor_temp.append(point_face_coor[k])
            k += 1
        one_point_face_coor_temp_arr = np.array(one_point_face_coor_temp)
        _, point_in_each_face_ids = random_point_sampling(one_point_face_coor_temp_arr, each_face_points_num)
        if i==0:
            face_points_selected_indices = point_in_each_face_ids
        else:
            face_points_selected_indices = np.concatenate((face_points_selected_indices, point_in_each_face_ids), axis=0)
        i += 1
    num_elements_to_remove = each_face_points_num*len(all_face_ids)-face_points_num
    if num_elements_to_remove>0:
        random_indices = np.random.choice(len(face_points_selected_indices), num_elements_to_remove, replace=False)
        face_points_selected_indices = np.delete(face_points_selected_indices, random_indices)

    sampled_face_points = np.zeros((face_points_num, face_points.shape[1]))
    sampled_edge_points = np.zeros((edge_points_num, edge_points.shape[1]))

    if edge_points_num <= edge_points.shape[0]:
        _, edge_points_selected_indices = random_point_sampling(point_edge_coor, edge_points_num)
    else:
        _, edge_points_selected_indices = np.concatenate(np.arange(0, point_edge_coor.shape[0]),
                                                         np.arange(0, edge_points_num - point_edge_coor.shape[0],
                                                                   dtype=int))

    i = 0
    for index in face_points_selected_indices:
        sampled_face_points[i] = face_points[index]
        i += 1
    j = 0
    for index in edge_points_selected_indices:
        sampled_edge_points[j] = edge_points[index]
        j += 1

    all_sampled_face_points, all_sampled_edge_points, point_instance_matrix,point_instance_label = data_argument_1(sampled_face_points, sampled_edge_points, get_matrix)

    return all_sampled_face_points, all_sampled_edge_points, point_instance_matrix, point_instance_label

def get_all_files_in_folder(folder_path):
    all_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    return files, all_files

def get_instance_label(sim_ids, face_ids):
    instance_label = []
    face_ids = face_ids.astype(int)
    face_ids=face_ids.tolist()
    for f_ids in face_ids:
        for i, ids in enumerate(sim_ids):
            if f_ids in ids:
                instance_label.append(i+1)
    assert len(instance_label)!=0
    return instance_label

def write_all_data_to_h5_v3(txt_file_path, h5_folder, face_points_num=1024, edge_points_num=256):
    _, allfiles = get_all_files_in_folder(txt_file_path)
    i = 0
    file_ids_t = []
    all_final_points_face_list_t = []
    all_final_points_edge_list_t = []
    all_face_id_list_t = []
    point_face_label_stand_list_t = []
    point_instance_label_list_t = []

    file_ids_v = []
    all_final_points_face_list_v = []
    all_final_points_edge_list_v = []
    all_face_id_list_v = []
    point_face_label_stand_list_v = []
    point_instance_label_list_v = []
    train_data_rate = 0.8
    random.seed(1997)
    random.shuffle(allfiles)
    for file in allfiles[:int(len(allfiles) * train_data_rate)]:
        files_name = file.strip().split("\\")[-1]
        file_id = int(files_name.split(".")[0][4:])
        face_points, edge_points = read_data_from_txt(file)
        all_sampled_face_points, all_sampled_edge_points, _, point_instance_label = sample_from_face_and_edge_points(face_points, edge_points,
                                                                                               face_points_num,
                                                                                               edge_points_num)

        assert len(all_sampled_edge_points) == len(all_sampled_edge_points)
        for j in range(len(all_sampled_face_points)):
            final_face_point, final_edge_point, point_face_id, point_face_label_stand = standardlize_point_cloud(
                all_sampled_face_points[j], all_sampled_edge_points[j])
            all_final_points_face_list_t.append(final_face_point)
            all_final_points_edge_list_t.append(final_edge_point)
            all_face_id_list_t.append(point_face_id)
            point_face_label_stand_list_t.append(point_face_label_stand)
            point_instance_label_list_t.append(point_instance_label)
            file_ids_t.append(file_id)

        i += 1
        if i % 50 == 0: print(i)

    for file in allfiles[int(len(allfiles) * train_data_rate):]:
        files_name = file.strip().split("\\")[-1]
        file_id = int(files_name.split(".")[0][4:])
        face_points, edge_points = read_data_from_txt(file)
        all_sampled_face_points, all_sampled_edge_points, _, point_instance_label = sample_from_face_and_edge_points(
            face_points, edge_points, face_points_num, edge_points_num)

        assert len(all_sampled_edge_points) == len(all_sampled_edge_points)
        for j in range(len(all_sampled_face_points)):
            final_face_point, final_edge_point, point_face_id, point_face_label_stand = standardlize_point_cloud(
                all_sampled_face_points[j], all_sampled_edge_points[j])
            all_final_points_face_list_v.append(final_face_point)
            all_final_points_edge_list_v.append(final_edge_point)
            all_face_id_list_v.append(point_face_id)
            point_face_label_stand_list_v.append(point_face_label_stand)
            point_instance_label_list_v.append(point_instance_label)
            file_ids_v.append(file_id)

        i += 1
        if i % 50 == 0: print(i)

    h5_path_t = os.path.join(h5_folder, "train_" + ".h5")
    save_h5(h5_path_t, file_ids_t, all_final_points_face_list_t, all_final_points_edge_list_t,
            all_face_id_list_t, class_label=point_face_label_stand_list_t, sim_ins_label = point_instance_label_list_t)

    h5_path_v = os.path.join(h5_folder, "validation_" + ".h5")
    save_h5(h5_path_v, file_ids_v, all_final_points_face_list_v, all_final_points_edge_list_v,
            all_face_id_list_v, class_label=point_face_label_stand_list_v, sim_ins_label = point_instance_label_list_v)

def read_origin_h5_write_modify_h5(origin_h5, new_h5):
    data_point_face, data_point_edge, data_part_id, data_class_label, data_instance_label, data_face_id = load_h5_for_ASIN_data(origin_h5)
    data_instance_label_new = np.zeros_like(data_instance_label)
    for k in range(data_instance_label.shape[0]):
        _, col_indices = np.where(data_class_label[k]==1)
        ins_labels_temp = (col_indices+1)*10+data_instance_label[k]
        unique_labels = np.unique(ins_labels_temp)

        ins_labels = np.zeros_like(data_instance_label[k])
        for i,label in enumerate(unique_labels):
            for j in range(ins_labels_temp.shape[0]):
                if ins_labels_temp[j] == label:
                    ins_labels[j] = i+1
        data_instance_label_new[k] = ins_labels
        if (k!=0)&(k%500==0):
            print(k)
    save_h5(new_h5, data_part_id, data_point_face, data_point_edge,
            data_face_id, class_label=data_class_label, sim_ins_label=data_instance_label_new)
