import torch
import torch.nn as nn
from torch.utils import data
from DataProcessing import Utils
import numpy as np
from Model_utils import PointASISTransformer_All_attribute, PointASISTransFormer_only_for_ins
from AverageMeter import AverageMeter
import time
import Builder
from torch.utils.tensorboard import SummaryWriter
import os
from Metric import Fine_tuning_Metric_for_sem
from Discriminative import DiscriminativeLoss
import random

def read_data(file, foe_attr=True):
    data_point_face, data_point_edge, data_part_id, data_class_label, data_instance_label, data_face_id = Utils.load_h5_for_ASIN_data(
        file)

    Face_points_tensor = torch.tensor(data_point_face, dtype=torch.float32)
    Edge_points_tensor = torch.tensor(data_point_edge, dtype=torch.float32)
    Part_id_tensor = torch.tensor(data_part_id, dtype=torch.float32)
    Class_label_tensor = torch.tensor(data_class_label, dtype=torch.float32)
    Instance_label_tensor = torch.tensor(data_instance_label, dtype=torch.float32)
    Data_id_tensor = torch.tensor(data_face_id, dtype=torch.float32)

    # concatenate the face points and edge points
    All_points = np.concatenate((Face_points_tensor, Edge_points_tensor), axis=1)
    All_points_tensor = torch.tensor(All_points, dtype=torch.float32)
    if foe_attr == False:
        return All_points_tensor, Face_points_tensor[:, :, :-2], Edge_points_tensor[:, :, :-4], Part_id_tensor, Class_label_tensor, Instance_label_tensor, Data_id_tensor
    return All_points_tensor, Face_points_tensor, Edge_points_tensor, Part_id_tensor, Class_label_tensor, Instance_label_tensor, Data_id_tensor

def read_AAG_data(file):
    points_attrs, points_face_ids, points_ins_labels, points_sem_labels = Utils.load_h5_for_AAG_attr_data(file)
    points_attrs = torch.tensor(points_attrs, dtype=torch.float32)
    points_face_ids = torch.tensor(points_face_ids, dtype=torch.float32)
    points_ins_labels = torch.tensor(points_ins_labels, dtype=torch.float32)
    points_sem_labels = torch.tensor(points_sem_labels, dtype=torch.float32)
    return points_attrs, points_face_ids, points_ins_labels, points_sem_labels

def data_builder_train(data_set, class_label_tensor, instance_label_tensor, Instance_label_num, batch_size):
    data_set = data.TensorDataset(data_set, class_label_tensor, instance_label_tensor, Instance_label_num)
    dataloader = data.DataLoader(dataset = data_set,
                                 batch_size=batch_size,
                                 shuffle = True,
                                 drop_last = True,
                                 )
    return dataloader

def data_builder_train_without_ins(data_set, class_label_tensor,batch_size):
    data_set = data.TensorDataset(data_set, class_label_tensor)
    dataloader = data.DataLoader(dataset = data_set,
                                 batch_size=batch_size,
                                 shuffle = True,
                                 drop_last = True,
                                 )
    return dataloader

def train_one_epoch(train_data_loader, fine_tuning_model, optimizer_type, max_epoch, epoch, writer, max_object_num, experiment_path,
                    lr=0.001, decay=0.05, device='cuda'):
    total_loss = 0
    total_loss_sem = 0
    total_loss_ins = 0
    total_acc_sem = 0
    total_acc_ins = 0

    epoch_start_time = time.time()
    losses = AverageMeter(['Loss'])
    num_iter = 0
    optimizer, schedule = Builder.build_opti_sche(fine_tuning_model, optimizer_type, decay, lr)
    for idx, (data, class_label, instance_label, object_num) in enumerate(train_data_loader):
        num_iter += 1
        points = data.to(device)
        B, N, _ = points.shape
        class_label = class_label.to(device)
        instance_label = instance_label.to(device)
        ret_ss, ret_isf = fine_tuning_model(points)
        class_label_fla = class_label.view(B * N, -1).to(float)
        ret_ss_fla = ret_ss.contiguous().view(B * N, -1)
        L_s = nn.CrossEntropyLoss()(class_label_fla, ret_ss_fla)
        Ins_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2)
        L_i = Ins_loss(ret_isf.unsqueeze(-1), instance_label.transpose(2, 1).unsqueeze(-1), object_num, max_object_num)

        loss = L_s + L_i
        try:
            loss.backward()
        except:
            loss = loss.mean()
            loss.backward()

        optimizer.step()
        fine_tuning_model.zero_grad()

        total_loss += loss.item()
        a_sem = Fine_tuning_Metric_for_sem(ret_ss, class_label)
        a_ins = 0

        total_loss_sem += L_s.item()
        total_loss_ins += L_i.item()
        total_acc_sem += a_sem
        total_acc_ins += a_ins

        losses.update([loss.item() * 1000])

    epoch_end_time = time.time()

    print(str(epoch)
          + ' time:' + str(epoch_end_time - epoch_start_time)
          + ' train loss:' + str("{:.3f}".format(total_loss / num_iter))
          + '\t' + 'train loss_sem:' + str("{:.3f}".format(total_loss_sem / num_iter))
          + '\t' + 'train loss_ins:' + str("{:.3f}".format(total_loss_ins / num_iter))
          + '\t' + 'train acc_sem:' + str("{:.3f}".format(total_acc_sem / num_iter))
          )
    writer.add_scalar('Train_Loss', total_loss / num_iter, epoch)
    writer.add_scalar('Train_loss_sem', total_loss_sem / num_iter, epoch)
    writer.add_scalar('Train_loss_ins', total_loss_ins / num_iter, epoch)
    writer.add_scalar('Train_acc_sem', total_acc_sem / num_iter, epoch)

    if epoch == max_epoch-1:
        Builder.save_checkpoint(fine_tuning_model, optimizer, epoch, task + f'ckpt-epoch-{epoch:03d}',
                                experiment_path)

def valid_one_epoch(valid_data_loader, fine_tuning_model, writer, epoch, max_object_num, device='cuda'):
    total_loss = 0
    total_loss_sem = 0
    total_loss_ins = 0
    total_acc_sem = 0
    total_acc_ins = 0

    epoch_start_time = time.time()
    losses = AverageMeter(['Loss'])
    num_iter = 0
    with torch.no_grad():
        for idx, (data, class_label, instance_label, object_num) in enumerate(valid_data_loader):
            num_iter += 1
            points = data.to(device)
            B, N, _ = points.shape
            class_label = class_label.to(device)
            instance_label = instance_label.to(device)
            ret_ss, ret_isf = fine_tuning_model(points)
            class_label_fla = class_label.view(B * N, -1).to(float)
            ret_ss_fla = ret_ss.contiguous().view(B * N, -1)
            L_s = nn.CrossEntropyLoss()(class_label_fla, ret_ss_fla)
            Ins_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2)
            L_i = Ins_loss(ret_isf.unsqueeze(-1), instance_label.transpose(2, 1).unsqueeze(-1), object_num,
                           max_object_num)
            if torch.isnan(L_i)==False:
                loss = L_i + L_s
                total_loss += loss.item()

                a_sem =Fine_tuning_Metric_for_sem(ret_ss, class_label)
                a_ins = 0

                total_loss_sem += L_s.item()
                total_loss_ins += L_i.item()
                total_acc_sem += a_sem
                total_acc_ins += a_ins

                losses.update([loss.item() * 1000])

    epoch_end_time = time.time()

    print(str(epoch)
          + ' time:' + str(epoch_end_time - epoch_start_time)
          + ' val loss:' + str("{:.3f}".format(total_loss / num_iter))
          + '\t' + 'val loss_sem:' + str("{:.3f}".format(total_loss_sem / num_iter))
          + '\t' + 'val loss_ins:' + str("{:.3f}".format(total_loss_ins / num_iter))
          + '\t' + 'val acc_sem:' + str("{:.3f}".format(total_acc_sem / num_iter))
          )
    writer.add_scalar('Val_Loss', total_loss / num_iter, epoch)
    writer.add_scalar('Val_loss_sem', total_loss_sem / num_iter, epoch)
    writer.add_scalar('Val_loss_ins', total_loss_ins / num_iter, epoch)
    writer.add_scalar('Val_acc_sem', total_acc_sem / num_iter, epoch)


def train_one_epoch_only_with_ins(train_data_loader, fine_tuning_model, optimizer_type, max_epoch, epoch, writer, max_object_num, experiment_path,
                    lr=0.001, decay=0.05, device='cuda'):
    total_loss = 0
    total_loss_ins = 0

    epoch_start_time = time.time()
    losses = AverageMeter(['Loss'])
    num_iter = 0
    optimizer, schedule = Builder.build_opti_sche(fine_tuning_model, optimizer_type, decay, lr)
    for idx, (data, class_label, instance_label, object_num) in enumerate(train_data_loader):
        num_iter += 1
        points = data.to(device)
        B, N, _ = points.shape
        instance_label = instance_label.to(device)
        ret_isf = fine_tuning_model(points)
        Ins_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2)
        L_i = Ins_loss(ret_isf.unsqueeze(-1), instance_label.transpose(2, 1).unsqueeze(-1), object_num, max_object_num)

        loss = L_i
        try:
            loss.backward()
        except:
            loss = loss.mean()
            loss.backward()

        optimizer.step()
        fine_tuning_model.zero_grad()
        total_loss += loss.item()
        total_loss_ins += L_i.item()

        losses.update([loss.item() * 1000])

    epoch_end_time = time.time()

    print(str(epoch)
          + ' time:' + str(epoch_end_time - epoch_start_time)
          + ' train loss:' + str("{:.3f}".format(total_loss / num_iter))
          + '\t' + 'train loss_ins:' + str("{:.3f}".format(total_loss_ins / num_iter))
          )
    writer.add_scalar('Train_Loss', total_loss / num_iter, epoch)
    writer.add_scalar('Train_loss_ins', total_loss_ins / num_iter, epoch)

    if epoch == max_epoch - 1:
        Builder.save_checkpoint(fine_tuning_model, optimizer, epoch, task + f'ckpt-epoch-{epoch:03d}',
                                experiment_path)

def valid_one_epoch_only_with_ins(valid_data_loader, fine_tuning_model, writer, epoch, max_object_num, device='cuda'):
    total_loss = 0
    total_loss_ins = 0
    total_acc_ins = 0

    epoch_start_time = time.time()
    losses = AverageMeter(['Loss'])
    num_iter = 0
    with torch.no_grad():
        for idx, (data, class_label, instance_label, object_num) in enumerate(valid_data_loader):
            num_iter += 1
            points = data.to(device)
            B, N, _ = points.shape
            instance_label = instance_label.to(device)
            ret_isf = fine_tuning_model(points)
            Ins_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2)
            L_i = Ins_loss(ret_isf.unsqueeze(-1), instance_label.transpose(2, 1).unsqueeze(-1), object_num,
                           max_object_num)
            if torch.isnan(L_i) == False:
                loss = L_i
                total_loss += loss.item()
                a_ins = 0

                total_loss_ins += L_i.item()
                total_acc_ins += a_ins

                losses.update([loss.item() * 1000])

    epoch_end_time = time.time()

    print(str(epoch)
          + ' time:' + str(epoch_end_time - epoch_start_time)
          + ' val loss:' + str("{:.3f}".format(total_loss / num_iter))
          + '\t' + 'val loss_ins:' + str("{:.3f}".format(total_loss_ins / num_iter))
          )
    writer.add_scalar('Val_Loss', total_loss / num_iter, epoch)
    writer.add_scalar('Val_loss_ins', total_loss_ins / num_iter, epoch)


def traning_only_with_ins(train_data_path, valid_data_path, max_epoch, logdir, task, ratio,ckpt_path, batch_size, experiment_path, device='cuda'):
    Face_points_tensor_t, _, Instance_label_tensor_t, Class_label_tensor_t = read_AAG_data(train_data_path)
    Face_points_tensor_v, _, Instance_label_tensor_v, Class_label_tensor_v = read_AAG_data(valid_data_path)
    num = Face_points_tensor_t.shape[0]

    Class_label_tensor_t_one_hot = torch.nn.functional.one_hot((Class_label_tensor_t).to(int))
    Class_label_tensor_v_one_hot = torch.nn.functional.one_hot((Class_label_tensor_v).to(int))
    _, _, class_num = Class_label_tensor_t_one_hot.shape

    Instance_label_tensor_t = Instance_label_tensor_t - 1
    Instance_label_tensor_v = Instance_label_tensor_v - 1
    Instance_label_tensor_t_one_hot = torch.nn.functional.one_hot((Instance_label_tensor_t).to(int))
    Instance_label_tensor_v_one_hot = torch.nn.functional.one_hot((Instance_label_tensor_v).to(int))
    max_object_num_t = torch.max(Instance_label_tensor_t).item() + 1
    max_object_num_v = torch.max(Instance_label_tensor_v).item() + 1

    random.seed(1998)
    index_num = torch.LongTensor(random.sample(range(num), int(num * ratio)))

    Face_points_tensor_t_ratio = Face_points_tensor_t[index_num]
    Class_label_tensor_t_ratio = Class_label_tensor_t_one_hot[index_num]
    Instance_label_tensor_t_one_hot_ratio = Instance_label_tensor_t_one_hot[index_num]
    Instance_label_tensor_t_ins_num_ratio = torch.max(Instance_label_tensor_t[index_num], dim=-1)[0] + 1

    train_dataloader = data_builder_train(Face_points_tensor_t_ratio, Class_label_tensor_t_ratio,
                                          Instance_label_tensor_t_one_hot_ratio,
                                          Instance_label_tensor_t_ins_num_ratio.to(int), batch_size)

    Instance_label_tensor_v_ins_num = torch.max(Instance_label_tensor_v, dim=-1)[0] + 1
    valid_dataloader = data_builder_train(Face_points_tensor_v, Class_label_tensor_v_one_hot,
                                          Instance_label_tensor_v_one_hot,
                                          Instance_label_tensor_v_ins_num.to(int), batch_size)

    base_model = PointASISTransFormer_only_for_ins(
        trans_dim=384,
        encoder_depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        encoder_dim=384,
        group_size=32,
        num_group=64,
        points_attribute=10,
        class_num=class_num,
        ins_out_dim=10,
    )

    base_model.load_model_from_ckpt(ckpt_path)
    base_model.to(device)
    # for param in base_model.encoder.parameters():
    #     param.requires_grad = False
    # for param in base_model.pos_embed.parameters():
    #     param.requires_grad = False
    # for param in base_model.blocks.parameters():
    #     param.requires_grad = False

    log_path = os.path.join(logdir, task)
    writer = SummaryWriter(log_path)

    for epoch in range(max_epoch):
        train_one_epoch_only_with_ins(train_dataloader,
                                      base_model,
                                      "AdamW",
                                      max_epoch,
                                      epoch,
                                      writer,
                                      max_object_num_t,
                                      experiment_path
                                      )

        if (epoch % 5 == 0) & (epoch != 0):
            valid_one_epoch_only_with_ins(valid_dataloader, base_model, writer, epoch, max_object_num_v)

def training_asis_all_attr(train_data_path, valid_data_path, max_epoch, logdir, task, ratio,ckpt_path, batch_size, experiment_path, device='cuda'):
    _, Face_points_tensor_t, _, _, Class_label_tensor_t, Instance_label_tensor_t, _ = read_data(train_data_path, False)
    _, Face_points_tensor_v, _, _, Class_label_tensor_v, Instance_label_tensor_v, _ = read_data(valid_data_path, False)
    num = Face_points_tensor_t.shape[0]

    Instance_label_tensor_t = Instance_label_tensor_t-1
    Instance_label_tensor_v = Instance_label_tensor_v-1
    Instance_label_tensor_t_one_hot = torch.nn.functional.one_hot((Instance_label_tensor_t).to(int))
    Instance_label_tensor_v_one_hot = torch.nn.functional.one_hot((Instance_label_tensor_v).to(int))
    max_object_num_t = torch.max(Instance_label_tensor_t).item() + 1
    max_object_num_v = torch.max(Instance_label_tensor_v).item() + 1

    random.seed(1998)
    index_num = torch.LongTensor(random.sample(range(num), int(num * ratio)))

    Face_points_tensor_t_ratio = Face_points_tensor_t[index_num]
    Class_label_tensor_t_ratio = Class_label_tensor_t[index_num]
    Instance_label_tensor_t_one_hot_ratio = Instance_label_tensor_t_one_hot[index_num]
    Instance_label_tensor_t_ins_num_ratio = torch.max(Instance_label_tensor_t[index_num], dim=-1)[0]+1

    train_dataloader = data_builder_train(Face_points_tensor_t_ratio, Class_label_tensor_t_ratio,
                                          Instance_label_tensor_t_one_hot_ratio, Instance_label_tensor_t_ins_num_ratio.to(int), batch_size)

    Instance_label_tensor_v_ins_num = torch.max(Instance_label_tensor_v, dim=-1)[0]+1
    valid_dataloader = data_builder_train(Face_points_tensor_v, Class_label_tensor_v, Instance_label_tensor_v_one_hot,
                                          Instance_label_tensor_v_ins_num.to(int), batch_size)


    base_model = PointASISTransformer_All_attribute(
        trans_dim=384,
        encoder_depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        encoder_dim=384,  # encoder_dim=trans_dim
        group_size=32,
        num_group=64, # 64
        points_attribute = 10,
    )

    base_model.load_model_from_ckpt(ckpt_path)
    base_model.to(device)
    log_path = os.path.join(logdir, task)
    writer = SummaryWriter(log_path)

    for epoch in range(max_epoch):
        train_one_epoch(train_dataloader,
                        base_model,
                        "AdamW",
                        max_epoch,
                        epoch,
                        writer,
                        max_object_num_t,
                        experiment_path,
                        )
        if (epoch % 5 == 0) & (epoch != 0):
            valid_one_epoch(valid_dataloader, base_model, writer, epoch, max_object_num_v)  # 40:num_group

if __name__ == "__main__":
    # Replace with your path
    train_data_path = r"F:\SSLFR_master\data\train.h5"
    valid_data_path = r"F:\SSLFR_master\data\validation.h5"
    logdir = r"F:\SSLFR_master\fine_tuning_logs"
    fine_tuning_experiment_path = r"F:\SSLFR_master\fine_tuning_experiment_path"
    task = "Fine_tuning_12_22"
    ckpt_path = r"F:\SSLFR_master\experiment_path\Pretrain_para.pth"

    s_time = time.time()
    training_asis_all_attr(train_data_path,
                        valid_data_path,
                        201,
                        logdir,
                        task,
                        0.08, #0.125, 0.25, 0.025, 0.0625, 0.5, 0.03, 1.0, 0.01
                        ckpt_path,
                        64,
                        fine_tuning_experiment_path,
                        )

###########################################For Training MFInstSeg###################################################################
    # train_data_path = r"F:\SSLFR_master\data\train_MFInstSeg_pt.h5"
    # valid_data_path = r"F:\SSLFR_master\data\valid_MFInstSeg_pt.h5"
    # logdir = r"F:\SSLFR_master\fine_tuning_logs"
    # fine_tuning_experiment_path = r"F:\SSLFR_master\fine_tuning_experiment_path"
    # task = "Fine_tuning_12_22_MFInstSeg"
    # ckpt_path = r"F:\SSLFR_master\experiment_path\Pretrain_para_MFInstSeg.pth"
    #
    # traning_only_with_ins(train_data_path,
    #                        valid_data_path,
    #                        201,
    #                        logdir,
    #                        task,
    #                        0.01,  # 0.125, 0.25, 0.025, 0.08, 0.5, 0.03, 1.0, 0.01
    #                        ckpt_path,
    #                        128,
    #                        fine_tuning_experiment_path,
    #                        )
