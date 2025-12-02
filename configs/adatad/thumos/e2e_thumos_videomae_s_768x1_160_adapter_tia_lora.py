_base_ = [
    "../../_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py",  # dataset config
    "../../_base_/models/actionformer.py",  # model config
]

window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input
dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),  # 减少解码线程，避免CPU过载
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),  # 减少解码线程，避免CPU过载
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),  # 减少解码线程，避免CPU过载
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)


model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerTIALoRA",  # 使用TIA Adapter + LoRA修正项
            img_size=224,
            patch_size=16,
            embed_dims=384,
            num_layers=12,  # 使用num_layers而不是depth
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,  # enable activation checkpointing
            total_frames=window_size * scale_factor,
            adapter_index=list(range(12)),  # 所有层都使用TIA LoRA Adapter
            # TIA LoRA Adapter配置
            adapter_kwargs=dict(
                mlp_ratio=0.25,  # 保持原始TIA adapter的mlp_ratio
                kernel_size=3,
                dilation=1,
                temporal_size=window_size * scale_factor,
                # LoRA参数
                lora_r=16,  # LoRA rank
                lora_alpha=32,  # LoRA alpha (缩放因子)
                lora_dropout=0.1,  # LoRA dropout
                lora_scale=1.0,  # LoRA修正项的缩放因子
            ),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=False,  # unfreeze the backbone
        ),
    ),
    projection=dict(
        in_channels=384,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
)

solver = dict(
    # 单GPU训练配置
    train=dict(batch_size=8, num_workers=8, prefetch_factor=4, persistent_workers=True),
    val=dict(batch_size=1, num_workers=2, prefetch_factor=2, persistent_workers=True),  # 减少验证时的batch_size和workers以避免OOM
    test=dict(batch_size=1, num_workers=1, prefetch_factor=1, persistent_workers=True),  # 进一步减少测试时的workers
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),  # adapter的原始Linear层（冻结）
            dict(name="lora", lr=2e-4, weight_decay=0.05),  # LoRA层
        ],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

work_dir = "exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter_tia_lora"

