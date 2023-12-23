import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models.layers import DropPath, trunc_normal_
import random

from Loss import ChamferDistance_only, MSE_CrossEntropy
from PointNet2_utils import PointNetFeaturePropagation_mod
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *


def load_pth(path):
    ckpt = torch.load(path)
    return ckpt

def farthest_point_sample_mod(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, A]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
        centroids_attrs: sampled pointcloud attributes, [B, npoint, A]
    """
    device = xyz.device
    B, N, A = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    centroids_attrs = torch.zeros(B, 1, A, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, A)
        centroid_coor = centroid[:, :, :3]
        if i == 0:
            centroids_attrs = centroid
        else:
            centroids_attrs = torch.concat((centroids_attrs, centroid), dim=1)

        dist = torch.sum((xyz[:, :, :3] - centroid_coor) ** 2, -1)
        dist = torch.tensor(dist,dtype=torch.float32)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids, centroids_attrs

class Group_for_all_attribute(nn.Module): #FPS + KNN
    def __init__(self, num_group, group_size, points_attribute):
        super(Group_for_all_attribute, self).__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.points_attribute = points_attribute

    def knn(self, center, xyz, group_size):
        dist = torch.cdist(center,xyz)
        neigbhors = dist.topk(k=group_size,dim=2,largest=False)
        return neigbhors.indices

    def forward(self, xyz):
        '''
            input: B N A
            ---------------------------
            output: B G M A
            center : B G A
        '''
        xyz_coors = xyz
        batch_size, num_points, _ = xyz.shape
        center_idx, centroids_attrs = farthest_point_sample_mod(xyz, self.num_group)
        centroids_coors =centroids_attrs[:, :, :3]

        idx = self.knn(centroids_attrs, xyz_coors, self.group_size) #B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, self.points_attribute).contiguous()
        # normalize
        pad_zeros = neighborhood[:,:,:,3:]
        neighborhood = neighborhood[:, :, :, :3] - centroids_coors.unsqueeze(2)
        neighborhood = torch.concat((neighborhood, pad_zeros), dim=-1)

        return neighborhood, center_idx, centroids_attrs, centroids_attrs[:,:,:3]

class Light_T_Net(nn.Module):
    def __init__(self):
        super(Light_T_Net, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)  # coordinate
        self.conv2 = nn.Conv1d(64, 512, 1)
        self.fc1 = nn.Linear(512, 36)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(512)
        self.k = 6

    def forward(self, x):
        bs = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = self.fc1(x)
        iden = Variable(torch.eye(self.k).flatten()).view(1, 36).repeat(
            bs, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 6, 6)
        return x

class Light_PointNet_Encoder(nn.Module):
    def __init__(self, points_attribute, encoder_channel=1024, transform=True):
        super(Light_PointNet_Encoder, self).__init__()
        self.encoder_channel = encoder_channel
        self.transform = transform
        self.points_attribute = points_attribute

        self.t_net = Light_T_Net()
        self.conv1 = nn.Conv1d(points_attribute, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, self.encoder_channel, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(self.encoder_channel)

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _= point_groups.shape
        point_groups = point_groups.reshape(bs*g, n, self.points_attribute) # number_of_attribute: 12 or 10

        if self.transform:
            point_groups_1 = point_groups[:, :, :6] # point coordinate and normal
            point_groups_2 = point_groups[:, :, 6:] # other attribute
            p_1 = self.t_net(point_groups_1.transpose(2, 1))
            p_1 = torch.bmm(point_groups_1, p_1)
            point_groups = torch.concat((p_1, point_groups_2), dim=2)

        p = F.relu(self.bn1(self.conv1(point_groups.transpose(2, 1))))
        p = F.relu(self.bn2(self.conv2(p)))
        p = F.relu(self.bn3(self.conv3(p)))
        p = self.bn4(self.conv4(p))

        p = torch.max(p, 2, keepdim=False)[0]

        return p.reshape(bs, g, self.encoder_channel)

class Light_PointNet_Encoder_for_mask(nn.Module):
    def __init__(self, points_attribute, encoder_channel=1024, transform=True):
        super(Light_PointNet_Encoder_for_mask, self).__init__()
        self.encoder_channel = encoder_channel
        self.transform = transform
        self.points_attribute = points_attribute

        self.t_net = Light_T_Net()
        self.conv1 = nn.Conv1d(points_attribute, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _= point_groups.shape
        point_groups = point_groups.reshape(bs*g, n, self.points_attribute) # number_of_attribute: 12 or 10

        if self.transform:
            point_groups_1 = point_groups[:, :, :6] # point coordinate and normal
            point_groups_2 = point_groups[:, :, 6:] # other attribute
            p_1 = self.t_net(point_groups_1.transpose(2, 1))
            p_1 = torch.bmm(point_groups_1, p_1)
            point_groups = torch.concat((p_1, point_groups_2), dim=2)

        p = F.relu(self.bn1(self.conv1(point_groups.transpose(2, 1))))
        p = self.bn2(self.conv2(p))

        p = p.transpose(2,1)
        return p.reshape(bs, g, n, 64)

# Transformers
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features # if out_features==None, then out_features = in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, A = x.shape #N is the number of groups.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, A//self.num_heads).permute(2, 0, 3, 1, 4) #3, B, num_heads, N, A/num_heads
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k, v: B, num_heads, N, A/num_heads
        attn = (q @ k.transpose(-2, -1)) * self.scale #B, num_heads, N, N
        attn = attn.softmax(dim=-1) #  B, num_heads, N, N
        attn = self.attn_drop(attn) # B, num_heads, N, N

        x = (attn @ v).transpose(1, 2) # B, N, num_heads, A/num_heads
        x = x.reshape(B, N, A) # B, N, A
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        '''
        Attention + MLP
        :param x:
        :return:
        '''
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

class TransformerEncoder_for_fine_tuning(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super(TransformerEncoder_for_fine_tuning, self).__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        feature_list = []
        fetch_dix = [3,7,11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_dix:
                feature_list.append(x)
        return feature_list

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super(TransformerDecoder, self).__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))
        return x

# Model
class MaskTransformer(nn.Module):
    def __init__(self, mask_ratio, trans_dim, depth, drop_path_rate, num_heads, encoder_dims, mask_type, points_attribute):
        super(MaskTransformer, self).__init__()
        self.mask_ratio = mask_ratio
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        self.encoder_dims = encoder_dims
        self.mask_type = mask_type

        self.encoder = Light_PointNet_Encoder(encoder_channel=self.encoder_dims, points_attribute=points_attribute, transform=False)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        ) # Only encode for coordinates

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)] # Increment dropout rate
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        center_coor = center[:,:,:3]
        for points in center_coor:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1) # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = torch.zeros([B, G]).to(center.device)
        for i in range(B):
            mask = torch.hstack([
                torch.zeros(G-self.num_mask),
                torch.ones(self.num_mask),
            ]).to(center.device)
            idx = torch.randperm(mask.nelement()).to(center.device)
            mask = mask[idx]
            overall_mask[i, :] = mask
        overall_mask = overall_mask.to(torch.bool)

        return overall_mask # B G

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood) # B,G,C
        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos

class MaskTransformer_Mod_All_attr(MaskTransformer):
    def __init__(self, mask_ratio, trans_dim, depth, drop_path_rate, num_heads, encoder_dims, mask_type, points_attribute):
        super(MaskTransformer_Mod_All_attr, self).__init__(mask_ratio, trans_dim, depth, drop_path_rate, num_heads, encoder_dims, mask_type, points_attribute)

        self.pos_embed = nn.Sequential(
            nn.Linear(points_attribute, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        ) # Only encode for coordinates

    def forward(self, neighborhood, center, bool_masked_pos, noaug=False):
        group_input_tokens = self.encoder(neighborhood)  # B,G,C
        batch_size, seq_len, C = group_input_tokens.size()
        A = neighborhood.shape[-1]
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, A)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis

class Double_Decoder_MAE_All_attribute(nn.Module):
    def __init__(self, mask_ratio, trans_dim, encoder_depth, drop_path_rate, num_heads, encoder_dim, mask_type,
                 group_size, num_group, decoder_depth, decoder_num_heads, points_attribute):
        super(Double_Decoder_MAE_All_attribute, self).__init__()

        self.MAE_encoder = MaskTransformer_Mod_All_attr(mask_ratio, trans_dim, encoder_depth, drop_path_rate, num_heads, encoder_dim, mask_type, points_attribute)
        self.trans_dim = trans_dim
        self.drop_path_rate = drop_path_rate
        self.points_attribute = points_attribute

        self.group_size = group_size
        self.num_group = num_group
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mask_ratio = mask_ratio

        self.loss_coor = 0
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.points_mask_feature = Light_PointNet_Encoder_for_mask(3, trans_dim, False)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(points_attribute, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder_coor = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.MAE_decoder_attr = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        self.group_divider = Group_for_all_attribute(num_group = self.num_group, group_size = self.group_size, points_attribute=points_attribute)
        self.predtion_head = nn.Sequential(
            nn.Conv1d(self.trans_dim+64, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        trunc_normal_(self.mask_token, std=0.02)

        self.build_loss_func()
        self.prediction_coor = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        self.prediction_nor = nn.Sequential(
            nn.Conv1d(128, 3, 1)
        )
        self.prediction_type = nn.Sequential(
            nn.Conv1d(128, 4, 1),
        )


    def build_loss_func(self):
        self.loss_func_coor = ChamferDistance_only().cuda()
        #self.loss_func_attr = Cosine_CrossEntropy().cuda()
        self.loss_func_attr = MSE_CrossEntropy().cuda()

    def _mask_center_rand(self, center, noaug=False):  # random mask
        '''
            center : B G A
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = torch.zeros([B, G]).to(center.device)
        for i in range(B):
            mask = torch.hstack([
                torch.zeros(G - self.num_mask),
                torch.ones(self.num_mask),
            ]).to(center.device)
            idx = torch.randperm(mask.nelement()).to(center.device)
            mask = mask[idx]
            overall_mask[i, :] = mask
        overall_mask = overall_mask.to(torch.bool)

        return overall_mask  # B G

    def forward(self,
                pts,
                vis=False,
                noaug=False,
                a = 0.4,# 0.375
                b = 0.3,# 0.25
                c = 0.3,# 0.25
                ):
        neighborhood, _, center_attrs, center = self.group_divider(pts)
        mask = self._mask_center_rand(center_attrs)
        x_vis= self.MAE_encoder(neighborhood, center_attrs, mask, noaug)

        _,N,G,_ = neighborhood.shape
        B, _, A = x_vis.shape  # A:attribute
        neighborhood_coor = neighborhood[:,:,:,:3]
        mask_ex = mask.unsqueeze(-1).expand(-1, -1, G)
        points_emd_mask = self.points_mask_feature(neighborhood_coor[mask_ex].reshape(B, -1, G, 3))
        pos_emd_vis = self.decoder_pos_embed(center_attrs[~mask]).reshape(B, -1, A)
        pos_emd_mask = self.decoder_pos_embed(center_attrs[mask]).reshape(B, -1, A)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        x_rec_coor = self.MAE_decoder_coor(x_full, pos_full, N)

        B, M, _ = x_rec_coor.shape
        # B,M,H -> B,H,M(for Conv) -> B,3*G,M -> B,M,12*G -> B*M,G,3
        rebuild_points_coor = self.prediction_coor(x_rec_coor.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)

        gt_points = neighborhood[mask].reshape(B * M, -1, self.points_attribute)
        gt_points_coor = neighborhood[mask].reshape(B * M, -1, self.points_attribute)[:, :, :3]
        gt_points_attr = neighborhood[mask].reshape(B * M, -1, self.points_attribute)[:,:,3:]
        loss_coor = self.loss_func_coor(rebuild_points_coor, gt_points_coor, bidirectional=True)

        x_rec_attr = self.MAE_decoder_attr(x_full, pos_full, N)
        x_rec_attrs = x_rec_attr.unsqueeze(2).expand(-1, -1, G, -1)
        x_rec_all_attr = torch.concat([points_emd_mask, x_rec_attrs], dim=-1).view(B * M, G, -1)
        x_all = self.predtion_head(x_rec_all_attr.transpose(2,1))

        rebuild_points_1 = self.prediction_nor(x_all).transpose(2, 1)
        rebuild_points_2 = nn.Softmax(dim=-2)(self.prediction_type(x_all)).transpose(2, 1)
        rebuild_points_attr = torch.concat((rebuild_points_1, rebuild_points_2), dim=-1)
        _, mseloss, cross_entropy_t = self.loss_func_attr(rebuild_points_attr,
                                                             gt_points_attr.reshape(B*M, G, -1), foe=False)
        cross_entropy_f = torch.tensor(0.)
        loss_all = a * loss_coor + b * mseloss + c * cross_entropy_t

        rebuild_points = torch.concat((rebuild_points_coor,rebuild_points_attr), dim=-1)
        rebuild_points_attr_with_gt = torch.concat((gt_points_coor,rebuild_points_attr), dim=-1)

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, self.points_attribute)

            center_coor_vis = center_attrs[~mask].unsqueeze(1)[:, :, :3]
            coor_vis = vis_points[:, :, :3] + center_coor_vis
            center_coor_mask = center_attrs[mask].unsqueeze(1)[:, :, :3]
            coor_mask = rebuild_points_coor + center_coor_mask

            coor_groud_truth = gt_points_coor + center_coor_mask

            zeros_center_coors_vis = center_attrs[~mask].unsqueeze(1)[:, :, 3:]
            zeros_center_coors_vis_pad = torch.zeros_like(zeros_center_coors_vis)
            center_coor_pad_vis = torch.concat((center_coor_vis, zeros_center_coors_vis_pad), dim=-1)
            attr_vis = vis_points+center_coor_pad_vis

            zeros_center_coors_mask = center_attrs[mask].unsqueeze(1)[:, :, 3:]
            zeros_center_coors_mask = torch.zeros_like(zeros_center_coors_mask)
            center_coor_pad_mask = torch.concat((center_coor_mask, zeros_center_coors_mask), dim=-1)
            attr_mask = rebuild_points_attr_with_gt + center_coor_pad_mask

            all_coors = torch.cat([coor_vis, coor_mask], dim=0)
            all_groudtruth_coors = torch.cat([coor_vis, coor_groud_truth], dim=0)
            all_attrs = torch.cat([attr_vis, attr_mask], dim=0)

            all_coors = all_coors.reshape(-1, 3).unsqueeze(0)
            all_attrs = all_attrs.reshape(-1, self.points_attribute).unsqueeze(0)
            coors_vis = coor_vis.reshape(-1, 3).unsqueeze(0)
            attr_vis = attr_vis.reshape(-1, self.points_attribute).unsqueeze(0)
            all_groudtruth_coors = all_groudtruth_coors.reshape(-1, 3).unsqueeze(0)
            return all_coors, all_attrs, coors_vis, attr_vis, all_groudtruth_coors
        else:
            return loss_all, gt_points, rebuild_points, loss_coor, mseloss, cross_entropy_t, cross_entropy_f

# Finetune model
class PointASISTransformer_All_attribute(nn.Module):
    def __init__(self,trans_dim, encoder_depth, drop_path_rate, num_heads, encoder_dim, group_size, num_group, points_attribute, class_num=4, ins_out_dim = 10):
        super(PointASISTransformer_All_attribute, self).__init__()
        self.trans_dim = trans_dim
        self.depth = encoder_depth
        self.drop_path_rate = drop_path_rate
        self.class_num = class_num
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dim
        self.ins_out_dim = ins_out_dim

        self.group_divider = Group_for_all_attribute(num_group=self.num_group, group_size=self.group_size, points_attribute=points_attribute)
        self.encoder = Light_PointNet_Encoder(points_attribute,  encoder_channel=self.encoder_dims,  transform=False)

        self.pos_embed = nn.Sequential(
            nn.Linear(points_attribute, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder_for_fine_tuning(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm=nn.LayerNorm(self.trans_dim)

        self.points_feature = Light_PointNet_Encoder_for_mask(points_attribute, trans_dim, False)

        self.PNFP_dim = 1024
        self.propagation_0 = PointNetFeaturePropagation_mod(in_channel=1152 + points_attribute,
                                                        mlp=[self.trans_dim * 4, self.PNFP_dim]) #1024

        self.semantic_segmentation_fine_head_1 = nn.Sequential(
            nn.Conv1d(self.PNFP_dim + self.trans_dim * 6, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.3),
        )

        self.instance_segmentation_fine_head_1 = nn.Sequential(
            nn.Conv1d(self.PNFP_dim + self.trans_dim * 6, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.3),
        )

        self.adaptation = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.3),
        )

        self.semantic_segmentation_fine_head_2 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),
            nn.Conv1d(128, self.class_num, 1),
        )

        self.instance_segmentation_fine_head_2 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),
            nn.Conv1d(128, self.ins_out_dim, 1),
            nn.BatchNorm1d(self.ins_out_dim),
        )

        self.sigmoid = nn.Sigmoid()

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        _, N, _ = pts.shape
        pts_attr = pts

        neighborhood, _, center_attrs, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        B, _, G, _ = neighborhood.shape
        pos = self.pos_embed(center_attrs)

        # transformer
        feature_list = self.blocks(group_input_tokens, pos)

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1152*2

        f_level_0 = self.propagation_0(pts_attr.transpose(2, 1), center_attrs.transpose(2, 1), pts_attr.transpose(2, 1), x)
        concat_f = torch.cat((f_level_0,x_global_feature), 1)

        ret_sem_1 = self.semantic_segmentation_fine_head_1(concat_f)
        ret_ins_1 = self.instance_segmentation_fine_head_1(concat_f)

        ret_sem = self.adaptation(ret_sem_1)
        ret_sins = ret_ins_1+ret_sem

        ret_ins_2 = self.instance_segmentation_fine_head_2(ret_sins)

        ret_isem = ret_sem_1
        ret_sem_2 = self.semantic_segmentation_fine_head_2(ret_isem) #

        ret_ss = nn.Softmax(dim=-2)(ret_sem_2).transpose(2, 1)
        ret_isf = ret_ins_2

        return ret_ss, ret_isf

# Finetune model for MFInstSeg dataset
class PointASISTransFormer_only_for_ins(PointASISTransformer_All_attribute):
    def __init__(self, trans_dim, encoder_depth, drop_path_rate, num_heads, encoder_dim, group_size, num_group,
                 points_attribute, class_num=4, ins_out_dim=10):
        super(PointASISTransFormer_only_for_ins, self).__init__(trans_dim, encoder_depth, drop_path_rate, num_heads,
                                                        encoder_dim, group_size, num_group, points_attribute, class_num,
                                                        ins_out_dim)
        self.PNFP_dim = 384
        self.propagation_1 = PointNetFeaturePropagation_mod(in_channel=384 + points_attribute,
                                                            mlp=[512, self.PNFP_dim])  # 384
        self.instance_segmentation_fine_head_1 = nn.Sequential(
            nn.Conv1d(self.PNFP_dim + 384, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.3),
        )

        self.instance_segmentation_fine_head_2 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),
            nn.Conv1d(128, self.ins_out_dim, 1),
            nn.BatchNorm1d(self.ins_out_dim),
        )

    def forward(self, pts):
        _, N, _ = pts.shape
        pts_attr = pts

        neighborhood, _, center_attrs, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        B, _, G, _ = neighborhood.shape
        pos = self.pos_embed(center_attrs)

        # transformer
        feature_list = self.blocks(group_input_tokens, pos)

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = feature_list[0]+feature_list[1]+feature_list[2]
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = x_max_feature + x_avg_feature

        f_level_0 = self.propagation_1(pts_attr.transpose(2, 1), center_attrs.transpose(2, 1), pts_attr.transpose(2, 1),
                                       x)
        concat_f = torch.cat((f_level_0, x_global_feature), 1)
        ret_ins_1 = self.instance_segmentation_fine_head_1(concat_f)
        ret_ins_2 = self.instance_segmentation_fine_head_2(ret_ins_1)

        ret_isf = ret_ins_2

        return ret_isf