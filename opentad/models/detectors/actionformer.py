# PyTorch 核心库
import torch
import torch.nn as nn

# 从上级目录导入模型注册器和基类
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
# 导入自定义的层组件
from ..bricks import Scale, AffineDropPath


# 使用装饰器注册模型，使其可以被配置文件识别和调用
@DETECTORS.register_module()
class ActionFormer(SingleStageDetector):
    """
    ActionFormer 模型类：用于视频动作检测的深度学习模型
    
    这是一个单阶段检测器，可以直接从视频特征中检测动作片段。
    主要包含特征提取、特征投影、特征融合和动作检测等功能。
    """
    def __init__(
        self,
        projection,      # 特征投影模块：将输入特征投影到模型需要的维度
        rpn_head,        # RPN（区域提议网络）头：用于生成动作检测的候选区域
        neck=None,       # 特征融合层（可选）：用于融合多尺度特征
        backbone=None,   # 骨干网络（可选）：用于提取视频特征
    ):
        """
        初始化 ActionFormer 模型
        
        Args:
            projection: 特征投影模块
            rpn_head: 区域提议网络头，负责生成动作检测结果
            neck: 可选的特征融合层
            backbone: 可选的骨干网络，用于特征提取
        """
        # 调用父类初始化方法，设置基础组件
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )

        # 获取多头注意力（Multi-Head Attention）的窗口大小
        n_mha_win_size = self.projection.n_mha_win_size
        # 如果窗口大小是整数，则将其扩展为列表（每个层使用相同的窗口大小）
        if isinstance(n_mha_win_size, int):
            # 窗口大小列表的长度 = 1 + projection 架构的最后一层索引
            self.mha_win_size = [n_mha_win_size] * (1 + projection.arch[-1])
        else:
            # 如果已经是列表，检查长度是否匹配
            assert len(n_mha_win_size) == (1 + projection.arch[-1])
            self.mha_win_size = n_mha_win_size
        
        # 获取最大序列长度（用于数据填充）
        self.max_seq_len = self.projection.max_seq_len

        # 计算最大整除因子，用于确保填充后的长度可以被步长整除
        max_div_factor = 1
        # 遍历 RPN 头的步长和注意力窗口大小
        for s, w in zip(rpn_head.prior_generator.strides, self.mha_win_size):
            # 计算实际步长：如果窗口大小 > 1，则步长需要考虑窗口大小
            stride = s * (w // 2) * 2 if w > 1 else s
            # 确保最大序列长度可以被步长整除（这是模型架构的要求）
            assert (
                self.max_seq_len % stride == 0
            ), f"max_seq_len {self.max_seq_len} must be divisible by fpn stride and window size {stride}"
            # 更新最大整除因子
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    def pad_data(self, inputs, masks):
        """
        对输入数据进行填充，使其长度符合模型要求
        
        这个方法确保输入特征的长度可以被模型处理，同时更新掩码以标记有效位置。
        
        Args:
            inputs: 输入特征张量，形状为 [batch_size, feature_dim, seq_len]
            masks: 掩码张量，标记哪些位置是有效的，形状为 [batch_size, seq_len]
        
        Returns:
            inputs: 填充后的特征张量
            pad_masks: 更新后的掩码张量
        """
        # 获取当前特征序列的长度（最后一个维度）
        feat_len = inputs.shape[-1]
        
        # 如果长度正好等于最大序列长度，直接返回
        if feat_len == self.max_seq_len:
            return inputs, masks
        # 如果长度小于最大序列长度，填充到最大序列长度
        elif feat_len < self.max_seq_len:
            max_len = self.max_seq_len
        else:  # 如果长度大于最大序列长度，需要填充到下一个可被整除的长度
            max_len = feat_len
            # 将输入填充到下一个可以被 max_div_factor 整除的大小
            stride = self.max_div_factor
            # 向上取整到最近的 stride 的倍数
            max_len = (max_len + (stride - 1)) // stride * stride

        # 计算需要填充的大小：[左侧填充, 右侧填充]
        padding_size = [0, max_len - feat_len]
        # 在序列末尾填充0值
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        # 创建新的掩码，初始化为全 False（表示无效位置）
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        # 将原始有效位置的掩码复制到新掩码中
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        """
        训练时的前向传播过程
        
        这个方法定义了模型在训练时的完整流程：
        1. 特征提取（可选）
        2. 数据填充
        3. 特征投影（可选）
        4. 特征融合（可选）
        5. 计算损失
        
        Args:
            inputs: 输入特征，形状为 [batch_size, feature_dim, seq_len]
            masks: 掩码，标记有效位置，形状为 [batch_size, seq_len]
            metas: 元数据信息
            gt_segments: 真实动作片段（ground truth segments），形状为 [batch_size, num_actions, 2]
            gt_labels: 真实动作标签（ground truth labels），形状为 [batch_size, num_actions]
            **kwargs: 其他可选参数
        
        Returns:
            losses: 损失字典，包含各种损失值和总损失 cost
        """
        # 初始化损失字典
        losses = dict()
        
        # 步骤1：如果配置了骨干网络，则通过骨干网络提取特征
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            # 否则直接使用输入特征
            x = inputs

        # 步骤2：对特征和掩码进行填充，使其长度符合模型要求
        x, masks = self.pad_data(x, masks)

        # 步骤3：如果配置了投影层，则对特征进行投影变换
        if self.with_projection:
            x, masks = self.projection(x, masks)

        # 步骤4：如果配置了特征融合层（neck），则进行多尺度特征融合
        if self.with_neck:
            x, masks = self.neck(x, masks)

        # 步骤5：通过 RPN 头进行前向传播，计算定位损失
        loc_losses = self.rpn_head.forward_train(
            x,
            masks,
            gt_segments=gt_segments,  # 真实动作片段
            gt_labels=gt_labels,      # 真实动作标签
            **kwargs,
        )
        # 将定位损失添加到总损失字典中
        losses.update(loc_losses)

        # 计算总损失：将所有损失值相加
        # 只有包含损失值的键才会被记录
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        """
        测试/推理时的前向传播过程
        
        这个方法定义了模型在测试时的完整流程，与训练流程类似，
        但最后返回的是检测结果而不是损失值。
        
        Args:
            inputs: 输入特征，形状为 [batch_size, feature_dim, seq_len]
            masks: 掩码，标记有效位置，形状为 [batch_size, seq_len]
            metas: 元数据信息（可选）
            infer_cfg: 推理配置（可选）
            **kwargs: 其他可选参数
        
        Returns:
            predictions: 预测结果元组，包含：
                - rpn_proposals: 检测到的动作片段提议，形状为 [batch_size, num_proposals, 2]
                - rpn_scores: 每个提议的置信度分数，形状为 [batch_size, num_proposals]
        """
        # 步骤1：如果配置了骨干网络，则通过骨干网络提取特征
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            # 否则直接使用输入特征
            x = inputs

        # 步骤2：对特征和掩码进行填充
        x, masks = self.pad_data(x, masks)

        # 步骤3：如果配置了投影层，则对特征进行投影变换
        if self.with_projection:
            x, masks = self.projection(x, masks)

        # 步骤4：如果配置了特征融合层，则进行多尺度特征融合
        if self.with_neck:
            x, masks = self.neck(x, masks)

        # 步骤5：通过 RPN 头进行前向传播，生成动作检测提议和分数
        rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)
        # 将提议和分数打包成元组返回
        predictions = rpn_proposals, rpn_scores
        return predictions

    def get_optim_groups(self, cfg):
        """
        将模型参数分为两组：应用权重衰减和不应用权重衰减
        
        这是优化器配置的重要步骤。不同的参数类型应该使用不同的权重衰减策略：
        - 权重参数（如 Linear、Conv1d）通常需要权重衰减来防止过拟合
        - 偏置参数和归一化层参数通常不需要权重衰减
        
        参考：https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        
        Args:
            cfg: 配置字典，包含 weight_decay 和 lr 等优化器参数
        
        Returns:
            optim_groups: 优化器参数组列表，包含两个组：
                - 第一组：需要权重衰减的参数
                - 第二组：不需要权重衰减的参数
        """
        # 初始化两个集合：分别存储需要和不需要权重衰减的参数名
        decay = set()      # 需要权重衰减的参数
        no_decay = set()   # 不需要权重衰减的参数
        
        # 定义白名单：这些模块的权重参数需要权重衰减
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        # 定义黑名单：这些模块的权重参数不需要权重衰减
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # 遍历模型中的所有模块和参数
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                # 构建完整的参数名称（模块名.参数名）
                fpn = "%s.%s" % (mn, pn) if mn else pn

                # 排除骨干网络的参数（通常骨干网络使用预训练权重，单独优化）
                if fpn.startswith("backbone"):
                    continue

                # 规则1：所有偏置（bias）参数都不需要权重衰减
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                # 规则2：白名单模块的权重参数需要权重衰减
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                # 规则3：黑名单模块（归一化层）的权重参数不需要权重衰减
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                # 规则4：自定义 Scale 层的 scale 参数不需要权重衰减
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    no_decay.add(fpn)
                # 规则5：相对位置编码（relative position encoding）不需要权重衰减
                elif pn.endswith("rel_pe"):
                    no_decay.add(fpn)

        # 验证：确保所有参数都被正确分类
        # 获取所有非骨干网络的参数
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        # 检查是否有参数同时出现在两个集合中（不应该发生）
        inter_params = decay & no_decay
        # 检查是否有参数没有被分类到任何一个集合中（不应该发生）
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # 创建 PyTorch 优化器参数组
        optim_groups = [
            {
                # 第一组：需要权重衰减的参数
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],  # 应用权重衰减
                "lr": cfg["lr"],                      # 学习率
            },
            {
                # 第二组：不需要权重衰减的参数
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,  # 不应用权重衰减
                "lr": cfg["lr"],      # 学习率
            },
        ]
        return optim_groups
