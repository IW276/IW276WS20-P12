# Authors: Chengchao Qu, Mickael Cormier

import torch
import torch.nn as nn


def c1d(in_channels, out_channels, kernel):
    if kernel > 1:
        padding = True
    else:
        padding = False
    conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding)
    bn = nn.BatchNorm1d(num_features=out_channels)
    lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
    return nn.Sequential(conv, bn, lrelu)


def block(in_channels, out_channels):
    b1 = c1d(in_channels, out_channels, 3)
    b2 = c1d(out_channels, out_channels, 3)
    return nn.Sequential(b1, b2)


def d1d(in_channels, out_channels):
    linear = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
    bn = nn.BatchNorm1d(num_features=out_channels)
    lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
    return nn.Sequential(linear, bn, lrelu)


class DDNet(nn.Module):
    def __init__(self, config):
        super(DDNet, self).__init__()
        self.config = config
        self.name = "ddnet"

        self.jcd_branch = nn.Sequential(
            c1d(config.feat_d, config.filters * 2, 1),
            nn.Dropout2d(0.1),
            c1d(config.filters * 2, config.filters, 3),
            nn.Dropout2d(0.1),
            c1d(config.filters, config.filters, 1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout2d(0.1)
        )

        self.slow_branch = nn.Sequential(
            c1d(config.joint_n * config.joint_d, config.filters * 2, 1),
            nn.Dropout2d(0.1),
            c1d(config.filters * 2, config.filters, 3),
            nn.Dropout2d(0.1),
            c1d(config.filters, config.filters, 1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout2d(0.1)
        )

        self.fast_branch = nn.Sequential(
            c1d(config.joint_n * config.joint_d, config.filters * 2, 1),
            nn.Dropout2d(0.1),
            c1d(config.filters * 2, config.filters, 3),
            nn.Dropout2d(0.1),
            c1d(config.filters, config.filters, 1),
            nn.Dropout2d(0.1)
        )

        self.gap_block = nn.Sequential(
            block(config.filters * 2, config.filters * 2),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            block(config.filters * 2, config.filters * 4),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            block(config.filters * 4, config.filters * 8),
            nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1)
        )

        self.linear_block = nn.Sequential(
            d1d(config.filters * 8, config.filters * 4),
            nn.Dropout(0.5),
            d1d(config.filters * 4, config.filters * 4),
            nn.Dropout(0.5),
            nn.Linear(in_features=config.filters * 4, out_features=self.config.clc_num, bias=True)
            # nn.Softmax() # No Softmax needed because of CrossEntropyLoss?
        )

    def forward(self, jcd, p_slow, p_fast):
        p = self.jcd_branch(jcd)
        x_d_slow = self.slow_branch(p_slow)
        # x_d_fast = self.fast_branch(p_fast)

        x = torch.cat((p, x_d_slow), dim=1)

        x = self.gap_block(x)

        x = x.squeeze(dim=2)

        x = self.linear_block(x)

        return x


class DDNetConfig:
    def __init__(self, n_classes, n_frames=6, n_joints=15, d_joints=2, n_filters=32):
        self.frame_l = n_frames  # Length of frames
        self.joint_n = n_joints  # Number of joints
        self.joint_d = d_joints  # Dimension of joints
        self.clc_num = n_classes  # Number of classes
        self.feat_d = n_joints * (n_joints - 1) // 2
        self.filters = n_filters


def main():
    cfg = DDNetConfig(n_classes=19)
    model = DDNet(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inp_jcd = torch.rand(1, cfg.feat_d, cfg.frame_l, device=device)
    inp_slow = torch.rand(1, cfg.joint_d * cfg.joint_n, cfg.frame_l, device=device)
    inp_fast = torch.rand(1, cfg.joint_d * cfg.joint_n, cfg.frame_l // 2, device=device)
    model.to(device).eval()
    model(inp_jcd, inp_slow, inp_fast)

    from torchsummaryX import summary
    summary(model.to(device).float(), inp_jcd, inp_slow, inp_fast)


if __name__ == '__main__':
    main()
