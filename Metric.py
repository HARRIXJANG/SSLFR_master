import torch

class Acc_Metric:
    def __init__(self, acc = 0.0):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return dict

def MAPE(yt, yp):
    L = torch.abs((yt - yp) / yt)
    L = torch.mean(L)
    return L

def Dis(yt, yp):
    x_t = yt[:, :, 0]
    y_t = yt[:, :, 1]
    z_t = yt[:, :, 2]
    x_p = yp[:, :, 0]
    y_p = yp[:, :, 1]
    z_p = yp[:, :, 2]
    d_x = torch.pow((x_p - x_t), 2)
    d_y = torch.pow((y_p - y_t), 2)
    d_z = torch.pow((z_p - z_t), 2)
    L = torch.mean(torch.pow((d_x+d_y+d_z), 1/2))
    return L

def Cosine(yt, yp):
    return torch.mean(torch.cosine_similarity(yt, yp, dim=-1))

def Mod_Metric(rebuild_points, gt_points):
    b, g, _ = gt_points.shape
    bg = b*g
    rp_coor = rebuild_points[:, :, :3]
    gt_coor = gt_points[:, :, :3]
    a1 = Dis(gt_coor, rp_coor).item()

    rp_nor = rebuild_points[:, :, 3:6]
    gt_nor = gt_points[:, :, 3:6]
    a_nor = Cosine(gt_nor, rp_nor).item()

    _, p_type_r = torch.max(rebuild_points[:, :, 6:10], dim=-1)
    _, p_type_g = torch.max(gt_points[:, :, 6:10], dim=-1)

    correct_1 = torch.eq(p_type_r, p_type_g)
    a2 = torch.sum(correct_1).item() / bg
    a3 = 0.
    return a1, a2, a3, a_nor

def Fine_tuning_Metric_for_sem(ret_ss, class_label):
    b, n, _ = class_label.shape
    _, ss_num = torch.max(ret_ss, dim=-1)
    _, cl_num = torch.max(class_label, dim=-1)
    correct_1 = torch.eq(ss_num, cl_num)
    acc_sem = torch.sum(correct_1).item() / (b*n)
    return acc_sem

