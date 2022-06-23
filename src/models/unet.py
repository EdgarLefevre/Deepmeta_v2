import torch  # type: ignore
import torch.nn as nn  # type: ignore

import src.models.unet_parts as up


class Unet(nn.Module):
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super(Unet, self).__init__()

        self.down1 = up.Down_Block(1, filters, conv_l=conv_l)
        self.down2 = up.Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = up.Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = up.Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.bridge = up.Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = up.Up_Block(filters * 16, filters * 8, drop_r, conv_l=conv_l)
        self.up2 = up.Up_Block(filters * 8, filters * 4, drop_r, conv_l=conv_l)
        self.up3 = up.Up_Block(filters * 4, filters * 2, drop_r, conv_l=conv_l)
        self.up4 = up.Up_Block(filters * 2, filters, drop_r, conv_l=conv_l)

        self.outc = up.OutConv(filters, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x = self.up1(bridge, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        return self.outc(x)


class Unet_res(nn.Module):
    """
    U-Net with residual connections
    """

    def __init__(self, filters: int, classes: int, drop_r: float = 0.1):
        """
        Initialize the U-Net model

        :param filters: number of filters in the first convolutional layer
        :type filters: int
        :param classes: number of classes to predict
        :type classes: int
        :param drop_r: dropout rate
        :type drop_r: float
        """
        super(Unet_res, self).__init__()

        self.down1 = up.Down_Block_res(1, filters)
        self.down2 = up.Down_Block_res(filters, filters * 2, drop_r)
        self.down3 = up.Down_Block_res(filters * 2, filters * 4, drop_r)
        self.down4 = up.Down_Block_res(filters * 4, filters * 8, drop_r)

        self.bridge = up.Bridge_res(filters * 8, filters * 16, drop_r)

        self.up1 = up.Up_Block_res(filters * 16, filters * 8, drop_r)
        self.up2 = up.Up_Block_res(filters * 8, filters * 4, drop_r)
        self.up3 = up.Up_Block_res(filters * 4, filters * 2, drop_r)
        self.up4 = up.Up_Block_res(filters * 2, filters, drop_r)

        self.outc = up.OutConv(filters, classes)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x = self.up1(bridge, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        return self.outc(x)


class Unet3plus(
    nn.Module
):  # todo: 160 dans cli (attention, voir dans les parts, mais 160 = 5*32)
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super(Unet3plus, self).__init__()

        self.down1 = up.Down_Block(1, filters, conv_l=conv_l)
        self.down2 = up.Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = up.Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = up.Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.bridge = up.Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = up.Up_Block_3p(filters * 16, 160, drop_r, conv_l=conv_l)
        self.up2 = up.Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up3 = up.Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up4 = up.Up_Block_3p(160, 160, drop_r, conv_l=conv_l)

        self.concat1 = up.Concat_Block(
            kernels_down=[8, 4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4, filters * 8],
            kernels_up=[],
            filters_up=[],
        )
        self.concat2 = up.Concat_Block(
            kernels_down=[4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4],
            kernels_up=[4],
            filters_up=[filters * 16],
        )
        self.concat3 = up.Concat_Block(
            kernels_down=[2, 1],
            filters_down=[filters, filters * 2],
            kernels_up=[4, 8],
            filters_up=[160, filters * 16],
        )
        self.concat4 = up.Concat_Block(
            kernels_down=[1],
            filters_down=[filters],
            kernels_up=[4, 8, 16],
            filters_up=[160, 160, filters * 16],
        )
        self.outc = up.OutConv(160, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x5 = self.up1(bridge, self.concat1([c1, c2, c3, c4], []))
        x6 = self.up2(x5, self.concat2([c1, c2, c3], [bridge]))
        x7 = self.up3(x6, self.concat3([c1, c2], [x5, bridge]))
        x8 = self.up4(x7, self.concat4([c1], [x6, x5, bridge]))
        return self.outc(x8)


class StridedUnet(nn.Module):
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super(StridedUnet, self).__init__()

        self.down1 = up.Strided_Down_Block(1, filters, conv_l=conv_l)
        self.down2 = up.Strided_Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = up.Strided_Down_Block(
            filters * 2, filters * 4, drop_r, conv_l=conv_l
        )
        self.down4 = up.Strided_Down_Block(
            filters * 4, filters * 8, drop_r, conv_l=conv_l
        )

        self.bridge = up.Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = up.Up_Block(filters * 16, filters * 8, drop_r, conv_l=conv_l)
        self.up2 = up.Up_Block(filters * 8, filters * 4, drop_r, conv_l=conv_l)
        self.up3 = up.Up_Block(filters * 4, filters * 2, drop_r, conv_l=conv_l)
        self.up4 = up.Up_Block(filters * 2, filters, drop_r, conv_l=conv_l)

        self.outc = up.OutConv(filters, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x = self.up1(bridge, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        return self.outc(x)


class URCNN(nn.Module):
    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        super(URCNN, self).__init__()
        self.dp = drop_r
        self.downsample = nn.MaxPool2d(2)
        self.down1 = up.RCL_Block(1, filters, 3, dp_rate=self.dp, conv_l=conv_l)
        self.down2 = up.RCL_Block(
            filters, filters * 2, 3, dp_rate=self.dp, conv_l=conv_l
        )
        self.down3 = up.RCL_Block(
            filters * 2, filters * 4, 3, dp_rate=self.dp, conv_l=conv_l
        )
        self.bridge = up.RCL_Block(
            filters * 4, filters * 8, 3, dp_rate=self.dp, conv_l=conv_l
        )
        self.upsample1 = nn.ConvTranspose2d(
            filters * 8, filters * 4, kernel_size=2, stride=2
        )
        self.up1 = up.RCL_Block(
            filters * 8, filters * 4, 3, dp_rate=self.dp, conv_l=conv_l
        )
        self.upsample2 = nn.ConvTranspose2d(
            filters * 4, filters * 2, kernel_size=2, stride=2
        )
        self.up2 = up.RCL_Block(
            filters * 4, filters * 2, 3, dp_rate=self.dp, conv_l=conv_l
        )
        self.upsample3 = nn.ConvTranspose2d(
            filters * 2, filters, kernel_size=2, stride=2
        )
        self.up3 = up.RCL_Block(filters * 2, filters, 3, dp_rate=self.dp, conv_l=conv_l)
        self.final_conv = nn.Sequential(
            conv_l(filters, filters, kernel_size=3, padding="same"),
            nn.BatchNorm2d(filters, momentum=0.9997),
            nn.ReLU(),
            nn.Dropout(self.dp),
            conv_l(filters, classes, kernel_size=1, padding="same"),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x1 = self.down1(x)
        x = self.downsample(x1)

        x2 = self.down2(x)
        x = self.downsample(x2)

        x3 = self.down3(x)
        x = self.downsample(x3)

        x = self.bridge(x)

        x = self.upsample1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)

        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)

        x = self.upsample3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)

        return self.final_conv(x)


class Att_Unet(nn.Module):
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super(Att_Unet, self).__init__()

        self.down1 = up.Down_Block(1, filters, conv_l=conv_l)
        self.down2 = up.Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = up.Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = up.Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.bridge = up.Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = up.Up_Block(filters * 16, filters * 8, drop_r, conv_l=conv_l)
        self.up2 = up.Up_Block(filters * 8, filters * 4, drop_r, conv_l=conv_l)
        self.up3 = up.Up_Block(filters * 4, filters * 2, drop_r, conv_l=conv_l)
        self.up4 = up.Up_Block(filters * 2, filters, drop_r, conv_l=conv_l)

        self.ag1 = up.Attention_Block(filters * 16, filters * 8)
        self.ag2 = up.Attention_Block(filters * 8, filters * 4)
        self.ag3 = up.Attention_Block(filters * 4, filters * 2)
        self.ag4 = up.Attention_Block(filters * 2, filters)

        self.outc = up.OutConv(filters, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x = self.up1(bridge, self.ag1(bridge, c4))
        x = self.up2(x, self.ag2(x, c3))
        x = self.up3(x, self.ag3(x, c2))
        x = self.up4(x, self.ag4(x, c1))
        return self.outc(x)


class Att_Unet3plus(
    nn.Module
):  # todo: 160 dans cli (attention, voir dans les parts, mais 160 = 5*32)
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super(Att_Unet3plus, self).__init__()

        self.down1 = up.Down_Block(1, filters, conv_l=conv_l)
        self.down2 = up.Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = up.Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = up.Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.bridge = up.Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = up.Up_Block_3p(filters * 16, 160, drop_r, conv_l=conv_l)
        self.up2 = up.Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up3 = up.Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up4 = up.Up_Block_3p(160, 160, drop_r, conv_l=conv_l)

        self.concat1 = up.Concat_Block(
            kernels_down=[8, 4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4, filters * 8],
            kernels_up=[],
            filters_up=[],
        )
        self.concat2 = up.Concat_Block(
            kernels_down=[4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4],
            kernels_up=[4],
            filters_up=[filters * 16],
        )
        self.concat3 = up.Concat_Block(
            kernels_down=[2, 1],
            filters_down=[filters, filters * 2],
            kernels_up=[4, 8],
            filters_up=[160, filters * 16],
        )
        self.concat4 = up.Concat_Block(
            kernels_down=[1],
            filters_down=[filters],
            kernels_up=[4, 8, 16],
            filters_up=[160, 160, filters * 16],
        )

        self.ag1 = up.Attention_Block(filters * 16, 128)
        self.ag2 = up.Attention_Block(160, 128)
        self.ag3 = up.Attention_Block(160, 128)
        self.ag4 = up.Attention_Block(160, 128)

        self.outc = up.OutConv(160, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x5 = self.up1(bridge, self.ag1(bridge, self.concat1([c1, c2, c3, c4], [])))
        x6 = self.up2(x5, self.ag2(x5, self.concat2([c1, c2, c3], [bridge])))
        x7 = self.up3(x6, self.ag3(x6, self.concat3([c1, c2], [x5, bridge])))
        x8 = self.up4(x7, self.ag4(x7, self.concat4([c1], [x6, x5, bridge])))
        return self.outc(x8)


class Unetpp(nn.Module):
    """
    U-Net++
    """

    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the U-Net++ model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super(Unetpp, self).__init__()

        self.down1 = up.Down_Block(1, filters, conv_l=conv_l)
        self.down2 = up.Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = up.Down_Block(filters * 2, filters * 4, drop_r, conv_l=conv_l)
        self.down4 = up.Down_Block(filters * 4, filters * 8, drop_r, conv_l=conv_l)

        self.nested01 = up.NestedBlock(filters * 2, filters, drop_r, conv_l=conv_l)
        self.nested02 = up.NestedBlock(filters * 3, filters, drop_r, conv_l=conv_l)
        self.nested03 = up.NestedBlock(filters * 4, filters, drop_r, conv_l=conv_l)

        self.nested11 = up.NestedBlock(filters * 4, filters * 2, drop_r, conv_l=conv_l)
        self.nested12 = up.NestedBlock(filters * 6, filters * 2, drop_r, conv_l=conv_l)

        self.nested21 = up.NestedBlock(filters * 8, filters * 4, drop_r, conv_l=conv_l)

        self.bridge = up.Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = up.Up_Block(filters * 16, filters * 8, drop_r, conv_l=conv_l)
        self.up2 = up.Up_Block(filters * 8, filters * 4, drop_r, conv_l=conv_l)
        self.up3 = up.Up_Block(filters * 4, filters * 2, drop_r, conv_l=conv_l)
        self.up4 = up.Up_Block(filters * 2, filters, drop_r, conv_l=conv_l)

        self.outc = up.OutConv(filters, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)

        x21 = self.nested21(c3, [], c4)

        x11 = self.nested11(c2, [], c3)
        x12 = self.nested12(x11, [c2], x21)

        x01 = self.nested01(c1, [], c2)
        x02 = self.nested02(x01, [c1], x11)
        x03 = self.nested03(x02, [c1, x01], x12)

        x = self.up1(bridge, c4)
        x = self.up2(x, x21)
        x = self.up3(x, x12)
        x = self.up4(x, x03)
        return self.outc(x)


class ResUnet3plus(
    nn.Module
):  # todo: 160 dans cli (attention, voir dans les parts, mais 160 = 5*32)
    """
    Implementation of the U-Net model from the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (Ronneberger et al., 2015)
    """

    def __init__(
        self,
        filters: int,
        classes: int,
        drop_r: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super(ResUnet3plus, self).__init__()

        self.down1 = up.Down_Block_res(1, filters)
        self.down2 = up.Down_Block_res(filters, filters * 2, drop_r)
        self.down3 = up.Down_Block_res(filters * 2, filters * 4, drop_r)
        self.down4 = up.Down_Block_res(filters * 4, filters * 8, drop_r)

        self.bridge = up.Bridge_res(filters * 8, filters * 16, drop_r)

        self.up1 = up.Up_Block_res_3p(filters * 16, 160, drop_r, conv_l=conv_l)
        self.up2 = up.Up_Block_res_3p(160, 160, drop_r, conv_l=conv_l)
        self.up3 = up.Up_Block_res_3p(160, 160, drop_r, conv_l=conv_l)
        self.up4 = up.Up_Block_res_3p(160, 160, drop_r, conv_l=conv_l)

        self.concat1 = up.Concat_Block(
            kernels_down=[8, 4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4, filters * 8],
            kernels_up=[],
            filters_up=[],
        )
        self.concat2 = up.Concat_Block(
            kernels_down=[4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4],
            kernels_up=[4],
            filters_up=[filters * 16],
        )
        self.concat3 = up.Concat_Block(
            kernels_down=[2, 1],
            filters_down=[filters, filters * 2],
            kernels_up=[4, 8],
            filters_up=[160, filters * 16],
        )
        self.concat4 = up.Concat_Block(
            kernels_down=[1],
            filters_down=[filters],
            kernels_up=[4, 8, 16],
            filters_up=[160, 160, filters * 16],
        )
        self.outc = up.OutConv(160, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x5 = self.up1(bridge, self.concat1([c1, c2, c3, c4], []))
        x6 = self.up2(x5, self.concat2([c1, c2, c3], [bridge]))
        x7 = self.up3(x6, self.concat3([c1, c2], [x5, bridge]))
        x8 = self.up4(x7, self.concat4([c1], [x6, x5, bridge]))
        return self.outc(x8)
