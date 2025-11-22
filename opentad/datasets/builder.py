import json
import torch
from torch.utils.data.dataloader import default_collate
from collections.abc import Sequence
from mmengine.registry import Registry, build_from_cfg, TRANSFORMS

DATASETS = Registry("dataset")
PIPELINES = TRANSFORMS


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset, batch_size, rank, world_size, shuffle=False, drop_last=False, **kwargs):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    assert batch_size % world_size == 0, f"batch size {batch_size} should be divided by world size {world_size}"

    # 从 kwargs 中获取 num_workers，用于条件设置
    num_workers = kwargs.get('num_workers', 0)

    # 构建 DataLoader 参数
    dataloader_kwargs = {
        'batch_size': batch_size // world_size,
        'collate_fn': collate,
        'pin_memory': True,
        'sampler': sampler,
    }

    # 只在 num_workers > 0 时设置 prefetch_factor 和 persistent_workers
    if num_workers > 0:
        # 64GB显存优化：增大 prefetch_factor 以提高数据加载速度，减少GPU等待
        dataloader_kwargs['prefetch_factor'] = 6
        # persistent_workers=False 以减少内存占用（之前因为OOM改过）
        dataloader_kwargs['persistent_workers'] = False

    # 合并用户传入的 kwargs（可能会覆盖上面的设置）
    dataloader_kwargs.update(kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        **dataloader_kwargs,
    )
    return dataloader


def collate(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    gpu_stack_keys = ["inputs", "masks"]

    collate_data = {}
    for key in batch[0]:
        if key in gpu_stack_keys:
            collate_data[key] = default_collate([sample[key] for sample in batch])
        else:
            collate_data[key] = [sample[key] for sample in batch]
    return collate_data


def get_class_index(gt_json_path, class_map_path):
    with open(gt_json_path, "r") as f:
        anno = json.load(f)

    anno = anno["database"]
    class_map = []
    for video_name in anno.keys():
        if "annotations" in anno[video_name]:
            for tmpp_data in anno[video_name]["annotations"]:
                if tmpp_data["label"] not in class_map:
                    class_map.append(tmpp_data["label"])

    class_map.sort()
    f2 = open(class_map_path, "w")
    for name in class_map:
        f2.write(name + "\n")
    f2.close()
    return class_map
