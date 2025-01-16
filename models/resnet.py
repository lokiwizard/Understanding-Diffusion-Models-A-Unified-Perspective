import torch
import torch.nn as nn

# 定义 3x3 卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

# 定义 1x1 卷积
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )

# 定义基本的残差块（用于 ResNet18 和 ResNet34）
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 第一个卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # 下采样层
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 第一层卷积 + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积 + BN
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

# 定义瓶颈残差块（用于 ResNet50 及以上）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # 第一个 1x1 卷积层
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 第二个 3x3 卷积层
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 第三个 1x1 卷积层
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        # 下采样层
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三个卷积块
        out = self.conv3(out)
        out = self.bn3(out)

        # 下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet 主体结构
class ResNet(nn.Module):

    def __init__(
        self, block, layers, num_classes=10,
        zero_init_residual=False, groups=1, width_per_group=64, return_features=False,
        replace_stride_with_dilation=None, norm_layer=None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # 输入通道数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 控制膨胀率
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.return_features = return_features

        # 初始卷积层
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 四个残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        # 自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 正态分布初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 常数初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 如果需要零初始化最后一个 BN 层
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 构建残差层
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # 如果需要膨胀
        if dilate:
            self.dilation *= stride
            stride = 1

        # 下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 第一个残差块
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            self.groups, self.base_width, previous_dilation, norm_layer
        ))
        self.inplanes = planes * block.expansion
        # 后续的残差块
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积和池化层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.return_features:
            return x


        # 如果是特征提取任务，输出形状为 [batch_size, C, H, W]
        # 如果是分类任务，输出形状为 [batch_size, num_classes]
        # 平均池化和全连接层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

# 定义 ResNet18 模型
def resnet18(num_classes=1000, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    return model

# 定义 ResNet34 模型
def resnet34(num_classes=1000, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return model

# 定义 ResNet50 模型
def resnet50(num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return model


# use case
if __name__ == '__main__':
    model = resnet18(num_classes=10, return_features=True)
    x = torch.randn(4, 3, 224, 224)
    print(model(x).shape)  # torch.Size([1, 512, 7, 7])

