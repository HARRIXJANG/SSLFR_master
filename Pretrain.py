import torch
from torch.utils import data
from DataProcessing import Utils
import numpy as np
from Model_utils import Double_Decoder_MAE_All_attribute
from AverageMeter import AverageMeter
import time
import Builder
from torch.utils.tensorboard import SummaryWriter
import os
from Metric import Mod_Metric

def read_data(training_file, foe_attr=True):
    Face_points, Edge_points, _, _, _,_ = Utils.load_h5_for_ASIN_data(training_file)
    Face_points_tensor = torch.tensor(Face_points, dtype=torch.float32)
    Edge_points_tensor = torch.tensor(Edge_points, dtype=torch.float32)
    # concatenate the face points and edge points
    All_points = np.concatenate((Face_points, Edge_points), axis=1)
    All_points_tensor = torch.tensor(All_points, dtype=torch.float32)
    if foe_attr == False:
        return All_points_tensor, Face_points_tensor[:, :, :-2], Edge_points_tensor[:, :, :-4]
    return All_points_tensor, Face_points_tensor, Edge_points_tensor

def read_data_for_coor_ASIN(training_file):
    Face_points, _, _, _, _, _ = Utils.load_h5_for_ASIN_data(training_file)
    Face_points_tensor = torch.tensor(Face_points, dtype=torch.float32)
    # concatenate the face points and edge points
    return Face_points_tensor[:, :, :3]

def read_data_for_attr(training_file):
    Face_points_attr, _, _, _ = Utils.load_h5_for_AAG_attr_data(training_file)
    Face_points_attr = torch.tensor(Face_points_attr, dtype=torch.float32)
    return Face_points_attr

def data_builder_train(data_set, batch_size):
    dataloader = data.DataLoader(dataset = data_set,
                                 batch_size=batch_size,
                                 shuffle = True,
                                 drop_last = True,
                                 )
    return dataloader

def train_net_double_decoder(train_data_loader, base_model, optimizer_type, epoch, max_epoch,
              experiment_path, writer, task, lr=0.001, decay=0.05, step_per_update=1, device='cuda'):
    base_model.train()

    total_loss = 0
    total_metric_coor = 0
    total_metric_nor = 0
    total_acc_t = 0
    total_acc_f = 0

    total_loss_t = 0
    total_loss_f = 0
    epoch_start_time = time.time()
    losses = AverageMeter(['Loss'])
    num_iter = 0
    optimizer, schedule = Builder.build_opti_sche(base_model, optimizer_type, decay, lr)
    for idx, (data) in enumerate(train_data_loader):
        num_iter += 1
        points = data.to(device)
        loss, gt_points, rebuild_points, chamferdistance, cosineloss, cross_entropy_t, cross_entropy_f = base_model(points)
        try:
            loss.backward()
        except:
            loss = loss.mean()
            loss.backward()

        if num_iter == step_per_update:
            optimizer.step()
            base_model.zero_grad()

        total_loss += loss.item()
        _, a_t, a_f, _ = Mod_Metric(rebuild_points, gt_points)
        total_metric_coor += chamferdistance.item()
        total_metric_nor += cosineloss.item()
        total_loss_t += cross_entropy_t.item()
        total_loss_f += cross_entropy_f.item()
        total_acc_t += a_t
        total_acc_f += a_f

        losses.update([loss.item()*1000])

    epoch_end_time = time.time()

    print(str(epoch)
          + ' time:' + str(epoch_end_time - epoch_start_time)
          + ' train loss:' + str("{:.3f}".format(total_loss / num_iter))
          + '\t' + 'train metric_c:' + str("{:.3f}".format(total_metric_coor / num_iter))
          + '\t' + 'train metric_n:' + str("{:.3f}".format(total_metric_nor / num_iter))
          + '\t' + 'train metric_t:' + str("{:.3f}".format(total_acc_t / num_iter))
          + '\t' + 'train loss_t:' + str("{:.3f}".format(total_loss_t / num_iter))
          )
    writer.add_scalar('Train_Loss', total_loss / num_iter, epoch)
    writer.add_scalar('Train_coor', total_metric_coor / num_iter, epoch)
    writer.add_scalar('Train_nor', total_metric_nor / num_iter, epoch)
    writer.add_scalar('Train_t', total_acc_t / num_iter, epoch)
    writer.add_scalar('Train_t_loss', total_loss_t / num_iter, epoch)


    if epoch == max_epoch-1:
        Builder.save_checkpoint(base_model, optimizer, epoch, task + f'ckpt-epoch-{epoch:03d}', experiment_path, logger=None)

def validate_net_double_decoder(validate_data_loader, base_model, writer, epoch, device='cuda'):
    base_model.eval()
    total_loss = 0
    total_metric_coor = 0
    total_metric_nor = 0
    total_acc_t = 0
    total_acc_f = 0

    total_loss_t = 0
    total_loss_f = 0
    with torch.no_grad():
        num_iter = 0
        for idx, (data) in enumerate(validate_data_loader):
            points = data.to(device)
            loss, gt_points, rebuild_points, chamferdistance, cosineloss, cross_entropy_t, cross_entropy_f = base_model(points, noaug=False)
            total_loss += loss.item()
            _, a_t, a_f, _ = Mod_Metric(rebuild_points, gt_points)
            total_metric_coor += chamferdistance.item()
            total_metric_nor += cosineloss.item()
            total_loss_t += cross_entropy_t.item()
            total_loss_f += cross_entropy_f.item()
            total_acc_t += a_t
            total_acc_f += a_f
            num_iter += 1
        assert num_iter!=0
        print('val loss:' + str(
            total_loss / num_iter)
              + '\t' + 'val coor:' + str("{:.3f}".format(total_metric_coor / num_iter))
              + '\t' + 'val nor:' + str("{:.3f}".format(total_metric_nor / num_iter))
              + '\t' + 'val metric_t:' + str("{:.3f}".format(total_acc_t / num_iter))
              + '\t' + 'val loss_t:' + str("{:.3f}".format(total_loss_t / num_iter))
              )
        writer.add_scalar('Val_Loss', total_loss / num_iter, epoch)
        writer.add_scalar('Val_coor', total_metric_coor / num_iter, epoch)
        writer.add_scalar('Val_nor', total_metric_nor / num_iter, epoch)
        writer.add_scalar('Val_t', total_acc_t / num_iter, epoch)
        writer.add_scalar('Val_loss_t', total_loss_t / num_iter, epoch)

def main(training_file, validation_file, batch_size, max_epoch, experiment_path, logdir, task, num_group, device='cuda'):
    _, pre_train_data, _ = read_data(training_file, foe_attr=False)
    pre_train_data_loader = data_builder_train(pre_train_data, batch_size)
    _, validate_data, _ = read_data(validation_file, foe_attr=False)
    pre_validate_data_loader = data_builder_train(validate_data, batch_size)
    base_model = Double_Decoder_MAE_All_attribute(mask_ratio=0.6,
                           trans_dim=384,
                           encoder_depth=12,
                           drop_path_rate=0.1,
                           num_heads=6,
                           encoder_dim=384,
                           mask_type='rand',
                           group_size=32,
                           num_group=num_group,
                           decoder_depth=4,
                           decoder_num_heads=6,
                           points_attribute=10)
    base_model.to(device)

    log_path = os.path.join(logdir, task)
    writer = SummaryWriter(log_path)

    for epoch in range(max_epoch):
        train_net_double_decoder(pre_train_data_loader,
                  base_model,
                  "AdamW",
                  epoch,
                  max_epoch,
                  experiment_path,
                  writer,
                  task,
                  )
        if (epoch % 5 == 0) & (epoch != 0):
            validate_net_double_decoder(pre_validate_data_loader, base_model, writer, epoch)  # 40:num_group

def main_for_AAGNet(training_file, validation_file,batch_size, max_epoch, experiment_path, logdir, task, num_group, device='cuda'):
    pre_train_data = read_data_for_attr(training_file)
    pre_train_data_loader = data_builder_train(pre_train_data, batch_size)
    validate_data = read_data_for_attr(validation_file)
    pre_validate_data_loader = data_builder_train(validate_data, batch_size)
    base_model = Double_Decoder_MAE_All_attribute(mask_ratio=0.6,
                                                  trans_dim=384,
                                                  encoder_depth=12,
                                                  drop_path_rate=0.1,
                                                  num_heads=6,
                                                  encoder_dim=384,
                                                  mask_type='rand',
                                                  group_size=32,
                                                  num_group=num_group,
                                                  decoder_depth=4,
                                                  decoder_num_heads=6,
                                                  points_attribute=10)
    base_model.to(device)
    log_path = os.path.join(logdir, task)
    writer = SummaryWriter(log_path)

    for epoch in range(max_epoch):
        train_net_double_decoder(pre_train_data_loader,
                  base_model,
                  "AdamW",
                  epoch,
                  max_epoch,
                  experiment_path,
                  writer,
                  task,
                  )
        if (epoch % 5 == 0) & (epoch != 0):
            validate_net_double_decoder(pre_validate_data_loader, base_model, writer, epoch)  # 40:num_group

if __name__ == "__main__":
    # Replace with your path
    training_file = r"F:\SSLFR_master\data\train.h5"
    val_file = r"F:\SSLFR_master\data\validation.h5"
    experiment_path = r"F:\SSLFR_master\experiment_path"
    log_dir = r"F:\SSLFR_master\logs"
    task = "pre_train_12_22"

    start_time = time.time()
    main(training_file,
         val_file,
         batch_size=256,
         max_epoch=101,
         experiment_path=experiment_path,
         logdir=log_dir,
         task=task,
         num_group=64)

    end_time = time.time()

    ###############################################For AAGNET#################################################
    # log_dir = r"F:\SSLFR_master\logs"
    # training_file = r"F:\SSLFR_master\data\train_MFInstSeg_pt.h5"
    # val_file = r"F:\SSLFR_master\data\valid_MFInstSeg_pt.h5"
    # experiment_path = r"F:\SSLFR_master\experiment_path"
    # task = "pre_train_12_22_MFInstSeg"
    #
    # start_time = time.time()
    # main_for_AAGNet(training_file,
    #                 val_file,
    #                 batch_size=256,
    #                 max_epoch=201,
    #                 experiment_path=experiment_path,
    #                 logdir=log_dir,
    #                 task=task,
    #                 num_group=64)
    #
    # end_time = time.time()
    # print(end_time-start_time)
