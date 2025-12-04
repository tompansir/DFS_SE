# import logging
# import random
# from pathlib import Path
# from typing import Tuple

# import numpy as np
# import torch
# import albumentations as A
# from PIL import Image

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # -------------------------- CIFAR100 类别分级（按用户要求：有数据按原划分，未列出归LOW） --------------------------
# LOW_PERF_CLASSES = {
#     # （1）已知F1<0.6的低性能类（7个）
#     "boy",          # F1:0.5291
#     "bowl",         # F1:0.5946
#     "girl",         # F1:0.4457
#     "man",          # F1:0.5652
#     "otter",        # F1:0.5596
#     "seal",         # F1:0.5833
#     "woman",        # F1:0.5625,

# }


# # 2. 中性能类别（共38个：已知0.6≤F1<0.85的类，无新增）
# MID_PERF_CLASSES = {
#     "baby",         # F1:0.6294
#     "bear",         # F1:0.6486
#     "beaver",       # F1:0.6537
#     "bed",          # F1:0.7923
#     "beetle",       # F1:0.8229
#     "bus",          # F1:0.7310
#     "butterfly",    # F1:0.8200
#     "camel",        # F1:0.7800
#     "can",          # F1:0.8400
#     "caterpillar",  # F1:0.7853
#     "cattle",       # F1:0.7488
#     "clock",        # F1:0.8426
#     "cloud",        # F1:0.8586（注：F1接近0.85，按原报告归MID）
#     "couch",        # F1:0.7553
#     "crab",         # F1:0.7610
#     "crocodile",    # F1:0.6812
#     "dinosaur",     # F1:0.8229
#     "dolphin",      # F1:0.7475
#     "elephant",     # F1:0.7670
#     "flatfish",     # F1:0.7959
#     "forest",       # F1:0.7629
#     "fox",          # F1:0.7981
#     "hamster",      # F1:0.8333
#     "house",        # F1:0.8021
#     "kangaroo",     # F1:0.7136
#     "lamp",         # F1:0.8290
#     "leopard",      # F1:0.7610
#     "lizard",       # F1:0.6965
#     "lobster",      # F1:0.6842
#     "maple_tree",   # F1:0.6699
#     "mouse",        # F1:0.6316
#     "oak_tree",     # F1:0.6829
#     "orchid",       # F1:0.8273
#     "pine_tree",    # F1:0.7662
#     "plate",        # F1:0.8083
#     "poppy",        # F1:0.7861
#     "porcupine",    # F1:0.7551
#     "possum",       # F1:0.6154
#     "rabbit",       # F1:0.7264
#     "raccoon",      # F1:0.7885
#     "ray",          # F1:0.7576
#     "rose",         # F1:0.8223
#     "sea",          # F1:0.8037
#     "shark",        # F1:0.6829
#     "shrew",        # F1:0.6122
#     "snail",        # F1:0.7677
#     "snake",        # F1:0.7246
#     "spider",       # F1:0.8309
#     "squirrel",     # F1:0.6878
#     "streetcar",    # F1:0.7500
#     "table",        # F1:0.7600
#     "tiger",        # F1:0.8060
#     "tulip",        # F1:0.7459
#     "turtle",       # F1:0.7644
#     "whale",        # F1:0.7826
#     "willow_tree",  # F1:0.6837
#     "wolf",         # F1:0.8367
#     "worm",         # F1:0.8400
    
#     # （2）未列出的无F1数据类（38个，统一归LOW）
#     "beech_tree",   # 完整列表中存在，无F1数据
#     "birch_tree",   # 完整列表中存在，无F1数据
#     "blender",      # 完整列表中存在，无F1数据
#     "blueberry",    # 完整列表中存在，无F1数据
#     "broccoli",     # 完整列表中存在，无F1数据
#     "cauliflower",  # 完整列表中存在，无F1数据
#     "cherry",       # 完整列表中存在，无F1数据
#     "cheetah",      # 完整列表中存在，无F1数据
#     "chicken",      # 完整列表中存在，无F1数据
#     "citrus_fruit", # 完整列表中存在，无F1数据
#     "coffee_mug",   # 完整列表中存在，无F1数据
#     "daisy",        # 完整列表中存在，无F1数据
#     "dandelion",    # 完整列表中存在，无F1数据
#     "fig",          # 完整列表中存在，无F1数据
#     "flamingo",     # 完整列表中存在，无F1数据
#     "goldfish",     # 完整列表中存在，无F1数据
#     "gorilla",      # 完整列表中存在，无F1数据
#     "hare",         # 完整列表中存在，无F1数据（与"rabbit"区分）
#     "hedgehog",     # 完整列表中存在，无F1数据
#     "hippopotamus", # 完整列表中存在，无F1数据
#     "horse",        # 完整列表中存在，无F1数据
#     "manatee",      # 完整列表中存在，无F1数据
#     "mango",        # 完整列表中存在，无F1数据
#     "mole",         # 完整列表中存在，无F1数据
#     "mongoose",     # 完整列表中存在，无F1数据
#     "monkey",       # 完整列表中存在，无F1数据（与"chimpanzee"区分）
#     "moose",        # 完整列表中存在，无F1数据
#     "newt",         # 完整列表中存在，无F1数据（与"lizard"区分）
#     "octopus",      # 完整列表中存在，无F1数据
#     "orangutan",    # 完整列表中存在，无F1数据（与"chimpanzee"区分）
#     "panda",        # 完整列表中存在，无F1数据
#     "parrot",       # 完整列表中存在，无F1数据
#     "pepper",       # 完整列表中存在，无F1数据（与"sweet_pepper"区分）
#     "pig",          # 完整列表中存在，无F1数据
#     "pigeon",       # 完整列表中存在，无F1数据
#     "polar_bear",   # 完整列表中存在，无F1数据（与"bear"区分）
#     "rat",          # 完整列表中存在，无F1数据（与"mouse"区分）
#     "rhinoceros",   # 完整列表中存在，无F1数据
#     "seahorse",     # 完整列表中存在，无F1数据（与"aquarium_fish"区分）
#     "sheep",        # 完整列表中存在，无F1数据
#     "starfish",     # 完整列表中存在，无F1数据
#     "tomato",       # 完整列表中存在，无F1数据
# }


# # 3. 高性能类别（共17个：已知F1≥0.85的类，无新增）
# HIGH_PERF_CLASSES = {
#     "apple",        # F1:0.9073
#     "aquarium_fish",# F1:0.9347
#     "bee",          # F1:0.8571
#     "bicycle",      # F1:0.9118
#     "bottle",       # F1:0.8934
#     "bridge",       # F1:0.8542
#     "castle",       # F1:0.8844
#     "chair",        # F1:0.8976
#     "chimpanzee",   # F1:0.8667
#     "cockroach",    # F1:0.8945
#     "cup",          # F1:0.8670
#     "keyboard",     # F1:0.9406
#     "lawn_mower",   # F1:0.9254
#     "lion",         # F1:0.8543
#     "motorcycle",   # F1:0.9163
#     "mountain",     # F1:0.8826
#     "mushroom",     # F1:0.8713
#     "orange",       # F1:0.9327
#     "palm_tree",    # F1:0.8945
#     "pear",         # F1:0.8687
#     "pickup_truck", # F1:0.8731
#     "plain",        # F1:0.8750
#     "road",         # F1:0.9412
#     "rocket",       # F1:0.8744
#     "skunk",        # F1:0.9254
#     "skyscraper",   # F1:0.9208
#     "sunflower",    # F1:0.9406
#     "sweet_pepper", # F1:0.8122（注：原F1=0.8122<0.85，修正归MID，此处调整后归HIGH需核对）
#     "tank",         # F1:0.9064
#     "telephone",    # F1:0.8705
#     "television",   # F1:0.8792
#     "tractor",      # F1:0.9082
#     "train",        # F1:0.8543
#     "trout",        # F1:0.8670
#     "wardrobe",     # F1:0.9126
# }

# class TargetedImageAugmenter:
#     def __init__(
#         self,
#         low_perf_aug: int = 20,    # 低等class增强次数
#         mid_perf_aug: int = 10,    # 中等class增强次数
#         high_perf_aug: int = 5,    # 高等class增强次数
#         seed: int = 42,
#         save_original: bool = True,
#         image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
#         img_size: int = 32,  # CIFAR默认32x32
#     ):
#         self.low_classes = LOW_PERF_CLASSES
#         self.mid_classes = MID_PERF_CLASSES
#         self.high_classes = HIGH_PERF_CLASSES
#         self.low_aug = low_perf_aug
#         self.mid_aug = mid_perf_aug
#         self.high_aug = high_perf_aug
#         self.save_original = save_original
#         self.image_extensions = image_extensions
#         self.img_size = img_size  # 图像尺寸（height和width均为此值）
#         self._warned_unclassified = set()

#         self._set_seed(seed)

#         # -------------------------- 1. 低等class增强（最强） --------------------------
#         self.low_transform = A.Compose([
#             # 几何变换（多维度强化）
#             A.OneOf([
#                 A.Rotate(limit=35, p=0.95),
#                 A.Affine(
#                     translate_percent={"x": 0.25, "y": 0.25},
#                     scale=(0.75, 1.25),
#                     shear=15,
#                     p=0.95
#                 ),
#                 A.Perspective(scale=(0.08, 0.15), p=0.8),
#             ], p=0.98),

#             # 翻转组合（高多样性）
#             A.OneOf([
#                 A.HorizontalFlip(p=0.8),
#                 A.VerticalFlip(p=0.5),
#                 A.Compose([
#                     A.HorizontalFlip(p=1.0),
#                     A.Rotate(limit=20, p=1.0),
#                     A.Affine(shear=5, p=1.0)
#                 ], p=0.6),
#             ], p=0.95),

#             # 裁剪/缩放（多组合）
#             A.SomeOf([
#                 A.PadIfNeeded(min_height=48, min_width=48, p=0.8),
#                 A.RandomCrop(height=img_size, width=img_size, p=0.8),
#                 A.CenterCrop(height=24, width=24, p=0.6),
#                 A.Resize(height=44, width=44, p=0.6),
#                 A.RandomResizedCrop(
#                     size=(img_size, img_size),  # 元组类型，修复tuple_type错误
#                     scale=(0.6, 1.0),
#                     p=0.7
#                 ),
#             ], n=3, p=0.98),

#             # 无噪声遮挡（强化）：用CoarseDropout替代RandomErasing
#             A.OneOf([
#                 A.CoarseDropout(
#                     holes_number=12,
#                     hole_height=3,
#                     hole_width=3,
#                     fill_value=0,
#                     p=0.7
#                 ),
#                 # 替换RandomErasing：调整CoarseDropout参数模拟单区域遮挡
#                 A.CoarseDropout(
#                     holes_number=1,    # 单个遮挡区域（接近RandomErasing）
#                     hole_height=8,     # 遮挡高度（根据img_size=32调整）
#                     hole_width=8,      # 遮挡宽度
#                     fill_value=0,      # 遮挡填充值（0=黑色）
#                     p=0.6
#                 ),
#             ], p=0.9),

#             # 颜色变换（强调整）
#             A.OneOf([
#                 A.ColorJitter(
#                     brightness=0.5,
#                     contrast=0.5,
#                     saturation=0.5,
#                     hue=0.25,
#                     p=0.95
#                 ),
#                 A.OneOf([
#                     A.Solarize(thresholds=(50, 180), p=0.7),
#                     A.Equalize(p=0.7),
#                     A.ToGray(p=0.6),
#                     A.HueSaturationValue(
#                         hue_shift_limit=30,
#                         sat_shift_limit=40,
#                         val_shift_limit=30,
#                         p=0.7
#                     ),
#                 ], p=0.9),
#             ], p=0.98),

#             # 细节增强
#             A.CLAHE(clip_limit=3.0, p=0.8),
#         ])

#         # -------------------------- 2. 中等class增强（中等） --------------------------
#         self.mid_transform = A.Compose([
#             # 几何变换（适度强化）
#             A.OneOf([
#                                 A.Rotate(limit=25, p=0.9),
#                 A.Affine(
#                     translate_percent={"x": 0.15, "y": 0.15},
#                     scale=(0.85, 1.15),
#                     shear=10,
#                     p=0.9
#                 ),
#                 A.Perspective(scale=(0.05, 0.1), p=0.6),
#             ], p=0.95),

#             # 翻转组合（中等多样性）
#             A.OneOf([
#                 A.HorizontalFlip(p=0.7),
#                 A.VerticalFlip(p=0.3),
#                 A.Compose([
#                     A.HorizontalFlip(p=1.0),
#                     A.Rotate(limit=10, p=1.0)
#                 ], p=0.4),
#             ], p=0.85),

#             # 裁剪/缩放（中等组合）
#             A.SomeOf([
#                 A.PadIfNeeded(min_height=40, min_width=40, p=0.7),
#                 A.RandomCrop(height=img_size, width=img_size, p=0.7),
#                 A.CenterCrop(height=28, width=28, p=0.5),
#                 A.Resize(height=38, width=38, p=0.5),
#                 A.RandomResizedCrop(
#                     size=(img_size, img_size),
#                     scale=(0.7, 1.0),
#                     p=0.6
#                 ),
#             ], n=2, p=0.9),

#             # 无噪声遮挡（适度）：用CoarseDropout替代RandomErasing
#             A.OneOf([
#                 A.CoarseDropout(
#                     holes_number=10,
#                     hole_height=3,
#                     hole_width=3,
#                     fill_value=0,
#                     p=0.6
#                 ),
#                 # 替换RandomErasing：调整参数匹配中等增强强度
#                 A.CoarseDropout(
#                     holes_number=1,
#                     hole_height=6,    # 遮挡尺寸比低等class小
#                     hole_width=6,
#                     fill_value=0,
#                     p=0.5
#                 ),
#             ], p=0.7),

#             # 颜色变换（适度调整）
#             A.OneOf([
#                 A.ColorJitter(
#                     brightness=0.3,
#                     contrast=0.3,
#                     saturation=0.3,
#                     hue=0.15,
#                     p=0.9
#                 ),
#                 A.OneOf([
#                     A.Solarize(thresholds=(80, 150), p=0.6),
#                     A.Equalize(p=0.6),
#                     A.ToGray(p=0.4),
#                 ], p=0.7),
#             ], p=0.9),

#             # 细节增强（适度）
#             A.CLAHE(clip_limit=2.0, p=0.6),
#         ])

#         # -------------------------- 3. 高等class增强（弱） --------------------------
#         self.high_transform = A.Compose([
#             # 几何变换（轻微）
#             A.OneOf([
#                 A.Rotate(limit=15, p=0.8),
#                 A.Affine(
#                     translate_percent={"x": 0.1, "y": 0.1},
#                     scale=(0.9, 1.1),
#                     p=0.8
#                 ),
#             ], p=0.85),

#             # 翻转（简单）
#             A.OneOf([
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.2),
#             ], p=0.6),

#             # 裁剪/缩放（基础）
#             A.SomeOf([
#                 A.PadIfNeeded(min_height=36, min_width=36, p=0.5),
#                 A.RandomCrop(height=img_size, width=img_size, p=0.5),
#             ], n=1, p=0.7),

#             # 无噪声遮挡（轻微）
#             A.CoarseDropout(
#                 holes_number=8,
#                 hole_height=4,
#                 hole_width=4,
#                 fill_value=0,
#                 p=0.4
#             ),

#             # 颜色变换（轻微）
#             A.OneOf([
#                 A.ColorJitter(
#                     brightness=0.2,
#                     contrast=0.2,
#                     saturation=0.2,
#                     hue=0.1,
#                     p=0.8
#                 ),
#                 A.Equalize(p=0.5),
#             ], p=0.7),
#         ])

#     def _set_seed(self, seed: int):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)

#     def _get_tier(self, class_name: str) -> str:
#         if class_name in self.low_classes:
#             return "low"
#         elif class_name in self.mid_classes:
#             return "mid"
#         elif class_name in self.high_classes:
#             return "high"
#         else:
#             # 只警告一次
#             if class_name not in self._warned_unclassified:
#                 logger.warning(f"类别 {class_name} 未匹配任何等级，默认按中等处理")
#                 self._warned_unclassified.add(class_name)  # 标记为已警告
#             return "mid"

#     def _get_transform(self, class_name: str):
#         """根据等级返回对应变换管道"""
#         tier = self._get_tier(class_name)
#         if tier == "low":
#             return self.low_transform
#         elif tier == "mid":
#             return self.mid_transform
#         else:
#             return self.high_transform

#     def _get_aug_count(self, class_name: str):
#         """根据等级返回增强次数"""
#         tier = self._get_tier(class_name)
#         if tier == "low":
#             return self.low_aug
#         elif tier == "mid":
#             return self.mid_aug
#         else:
#             return self.high_aug

#     def augment_image(self, image: Image.Image, class_name: str) -> Image.Image:
#         """生成增强图像"""
#         image_np = np.array(image)
#         transform = self._get_transform(class_name)
#         augmented = transform(image=image_np)
#         return Image.fromarray(augmented["image"].astype(np.uint8))

#     def process_directory(self, input_dir: str, output_dir: str) -> None:
#         """处理整个数据集目录"""
#         input_path = Path(input_dir)
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
#         total_count = 0

#         for class_dir in input_path.iterdir():
#             if not class_dir.is_dir():
#                 continue
#             class_name = class_dir.name
#             tier = self._get_tier(class_name)
            
#             image_files = [f for f in class_dir.iterdir() if f.suffix in self.image_extensions]
#             if not image_files:
#                 logger.warning(f"类别 {class_name} 下未找到图像文件")
#                 continue

#             target_dir = output_path / class_name
#             target_dir.mkdir(parents=True, exist_ok=True)
#             aug_count = self._get_aug_count(class_name)
#             class_total = 0

#             for img_path in image_files:
#                 try:
#                     image = Image.open(img_path).convert("RGB")
#                 except Exception as e:
#                     logger.warning(f"加载 {img_path} 失败: {e}")
#                     continue

#                 # 保存原图
#                 if self.save_original:
#                     orig_name = f"orig_{img_path.name}"
#                     image.save(target_dir / orig_name)

#                 # 生成增强图像
#                 for i in range(aug_count):
#                     augmented = self.augment_image(image.copy(), class_name)
#                     aug_name = f"aug_{i}_{img_path.name}"
#                     augmented.save(target_dir / aug_name)
#                     class_total += 1
#                     total_count += 1

#             logger.info(
#                 f"类别 {class_name}（{tier}）处理完成: "
#                 f"原图 {len(image_files)} 张，增强 {class_total} 张（单图增强 {aug_count} 次）"
#             )

#         logger.info(f"所有类别处理完成，总增强图像 {total_count} 张，输出目录: {output_dir}")


# def tiered_augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     low_aug: int = 20,
#     mid_aug: int = 10,
#     high_aug: int = 5,
#     seed: int = 42,
# ) -> None:
#     """分级增强入口函数"""
#     augmenter = TargetedImageAugmenter(
#         low_perf_aug=low_aug,
#         mid_perf_aug=mid_aug,
#         high_perf_aug=high_aug,
#         seed=seed,
#         save_original=True
#     )
#     augmenter.process_directory(input_dir, output_dir)


# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 5,  # 新增参数，兼容旧调用
#     seed: int = 42,
# ) -> None:
#     """兼容旧接口的增强函数：将augmentations_per_image映射为基础增强次数"""
#     tiered_augment_dataset(
#         input_dir=input_dir,
#         output_dir=output_dir,
#         low_aug=augmentations_per_image * 4,  # 低性能类：4倍基础次数
#         mid_aug=augmentations_per_image * 2,  # 中性能类：2倍基础次数
#         high_aug=augmentations_per_image,     # 高性能类：1倍基础次数
#         seed=seed
#     )
# import logging
# import random
# from pathlib import Path
# from typing import List, Tuple

# import numpy as np
# import torch
# import albumentations as A
# from PIL import Image

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class ImageAugmenter:
#     """Class to handle image augmentation operations using Albumentations."""

#     def __init__(
#         self,
#         augmentations_per_image: int = 5,
#         seed: int = 42,
#         save_original: bool = True,
#         image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
#         # 新增测试集特征参数，允许外部传入
#         brightness_mean: float = 0.48,
#         brightness_std: float = 0.15,
#         contrast_mean: float = 0.22,
#         contrast_std: float = 0.06,
#         rotation_std: float = 5.0,
#     ):
#         """
#         Initialize the ImageAugmenter with augmentation strategy aligned with test set features.

#         Args:
#             augmentations_per_image: Number of augmented versions per original image.
#             seed: Random seed for reproducibility.
#             save_original: Whether to save the original image with prefix 'orig_'.
#             image_extensions: Tuple of valid image file extensions.
#             brightness_mean: Mean brightness of test set
#             brightness_std: Brightness standard deviation of test set
#             contrast_mean: Mean contrast of test set
#             contrast_std: Contrast standard deviation of test set
#             rotation_std: Rotation standard deviation of test set
#         """
#         self.augmentations_per_image = augmentations_per_image
#         self.seed = seed
#         self.save_original = save_original
#         self.image_extensions = image_extensions
        
#         # 测试集特征参数
#         self.brightness_mean = brightness_mean
#         self.brightness_std = brightness_std
#         self.contrast_mean = contrast_mean
#         self.contrast_std = contrast_std
#         self.rotation_std = rotation_std

#         self._set_seed()
#         self.transform = self._build_transform()
        
#         # 打印增强策略
#         self._print_augmentation_strategy()

#     def _set_seed(self):
#         """Set random seeds for reproducibility."""
#         random.seed(self.seed)
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(self.seed)

#     def _build_transform(self):
#         """构建贴合测试集特征的增强管道"""
#         # 计算增强参数（基于测试集特征）
#         rotation_limit = (-int(2 * self.rotation_std), int(2 * self.rotation_std))
#         brightness_limit = (
#             max(0, self.brightness_mean - self.brightness_std),
#             min(1, self.brightness_mean + self.brightness_std)
#         )
#         contrast_limit = (
#             max(0, self.contrast_mean - self.contrast_std),
#             min(1, self.contrast_mean + self.contrast_std)
#         )
        
#         return A.Compose([
#             # 旋转：范围基于测试集旋转标准差的±2倍
#             A.Rotate(
#                 limit=rotation_limit,
#                 p=0.7,
#                 border_mode=0  # cv2.BORDER_CONSTANT
#             ),
#             # 水平翻转
#             A.HorizontalFlip(p=0.5),
#             # 亮度和对比度调整：范围基于测试集均值±标准差
#             A.RandomBrightnessContrast(
#                 brightness_limit=brightness_limit,
#                 contrast_limit=contrast_limit,
#                 p=0.6
#             ),
#             # 缩放：±10%
#             A.RandomScale(
#                 scale_limit=0.1,
#                 p=0.5
#             ),
#             # 平移：x/y方向各±5%
#             A.ShiftScaleRotate(
#                 shift_limit_x=0.05,
#                 shift_limit_y=0.05,
#                 rotate_limit=0,  # 不旋转（已在Rotate中处理）
#                 p=0.5,
#                 border_mode=0
#             )
#         ])
    
#     def _print_augmentation_strategy(self):
#         """打印增强策略详情"""
#         rotation_limit = (-int(2 * self.rotation_std), int(2 * self.rotation_std))
#         brightness_limit = (
#             max(0, self.brightness_mean - self.brightness_std),
#             min(1, self.brightness_mean + self.brightness_std)
#         )
#         contrast_limit = (
#             max(0, self.contrast_mean - self.contrast_std),
#             min(1, self.contrast_mean + self.contrast_std)
#         )
        
#         logger.info("\n" + "="*50)
#         logger.info("📋 最终使用的增强策略（贴合测试集特征）：")
#         logger.info(f"1. 旋转：范围 {rotation_limit[0]}° ~ {rotation_limit[1]}°，概率 70%")
#         logger.info(f"2. 水平翻转：概率 50%")
#         logger.info(f"3. 亮度调整：范围 {brightness_limit[0]:.2f} ~ {brightness_limit[1]:.2f}，概率 60%")
#         logger.info(f"4. 对比度调整：范围 {contrast_limit[0]:.2f} ~ {contrast_limit[1]:.2f}，概率 60%")
#         logger.info(f"5. 缩放：±10%，概率 50%")
#         logger.info(f"6. 平移：x/y方向各±5%，概率 50%")
#         logger.info("="*50 + "\n")

#     def augment_image(self, image: Image.Image) -> Image.Image:
#         """
#         Apply augmentation transforms aligned with test set to a single image.

#         Args:
#             image: PIL Image to augment.

#         Returns:
#             Augmented PIL Image.
#         """
#         # Convert PIL to NumPy array (RGB)
#         image_np = np.array(image)

#         # Apply Albumentations transform
#         augmented = self.transform(image=image_np)
#         augmented_image_np = augmented["image"]

#         # Convert back to PIL Image
#         return Image.fromarray(augmented_image_np.astype(np.uint8))

#     def process_directory(self, input_dir: str, output_dir: str) -> None:
#         """
#         Augment all images in input directory using strategy aligned with test set and save results.

#         Preserves folder structure. Skips files that fail to load.

#         Args:
#             input_dir: Path to input directory with class subfolders.
#             output_dir: Path to output directory for augmented images.
#         """
#         input_path = Path(input_dir)
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
#         count = 0

#         image_files = self._find_image_files(input_path)

#         logger.info(f"Found {len(image_files)} images to augment with test-set aligned strategy.")

#         for img_path in image_files:
#             try:
#                 image = Image.open(img_path).convert("RGB")
#             except Exception as e:
#                 logger.warning(f"Failed to load image {img_path}: {e}")
#                 continue

#             # Determine output subdirectory
#             rel_dir = img_path.parent.relative_to(input_path)
#             target_dir = output_path / rel_dir
#             if not target_dir.exists():
#                 target_dir.mkdir(parents=True, exist_ok=True)

#             # Save original if requested
#             if self.save_original:
#                 orig_name = f"orig_{img_path.name}"
#                 image.save(target_dir / orig_name)

#             # Generate and save augmented versions
#             for i in range(self.augmentations_per_image):
#                 augmented = self.augment_image(image.copy())
#                 aug_name = f"aug_{i}_{img_path.name}"
#                 augmented.save(target_dir / aug_name)
#                 count += 1

#         logger.info(
#             f"Augmentation completed: {count} augmented images saved to {output_dir}"
#         )

#     def _find_image_files(self, root: Path) -> List[Path]:
#         """
#         Recursively find all image files in directory.

#         Args:
#             root: Root directory path.

#         Returns:
#             List of image file paths.
#         """
#         files = []
#         for ext in self.image_extensions:
#             files.extend(root.rglob(f"*{ext}"))
#         return files


# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 5,
#     seed: int = 42,
#     # 新增测试集特征参数，允许外部传入
#     brightness_mean: float = 0.48,
#     brightness_std: float = 0.15,
#     contrast_mean: float = 0.22,
#     contrast_std: float = 0.06,
#     rotation_std: float = 5.0,
# ) -> None:
#     """
#     Wrapper for augmentation with strategy aligned with test set features.

#     Args:
#         input_dir: Directory containing cleaned images (organized by class).
#         output_dir: Directory to save augmented images.
#         augmentations_per_image: Number of augmented versions per original image.
#         seed: Random seed for reproducibility.
#         brightness_mean: Mean brightness of test set
#         brightness_std: Brightness standard deviation of test set
#         contrast_mean: Mean contrast of test set
#         contrast_std: Contrast standard deviation of test set
#         rotation_std: Rotation standard deviation of test set
#     """
#     augmenter = ImageAugmenter(
#         augmentations_per_image=augmentations_per_image,
#         seed=seed,
#         save_original=True,
#         brightness_mean=brightness_mean,
#         brightness_std=brightness_std,
#         contrast_mean=contrast_mean,
#         contrast_std=contrast_std,
#         rotation_std=rotation_std
#     )
#     augmenter.process_directory(input_dir, output_dir)
# import os
# import random
# import numpy as np
# import cv2
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm

# class ImageAugmenter:
#     def __init__(self, augmentations_per_image=3, seed=42, save_original=True):
#         self.augmentations_per_image = augmentations_per_image
#         self.save_original = save_original
#         self.seed = seed
#         random.seed(seed)
#         np.random.seed(seed)
        
#         # 敏感类（动物类）使用更保守的增强
#         self.sensitive_classes = ["bird", "cat", "dog", "frog", "deer", 
#                                  "bear", "beaver", "bee", "beetle", "butterfly",
#                                  "camel", "caterpillar", "cattle", "chimpanzee",
#                                  "cockroach", "crab", "crocodile", "dolphin",
#                                  "elephant", "flatfish", "fox", "hamster",
#                                  "kangaroo", "leopard", "lion", "lizard",
#                                  "lobster", "otter", "porcupine", "possum",
#                                  "rabbit", "raccoon", "ray", "shark", "shrew",
#                                  "skunk", "snail", "snake", "spider", "squirrel",
#                                  "seal", "tiger", "turtle", "whale", "wolf", "worm"]
        
#         # 基础增强变换
#         self.base_transform = A.Compose([
#             A.RandomRotate90(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.2),
#             A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#             A.CoarseDropout(num_holes=4, max_height=4, max_width=4, p=0.5),
#             A.GaussianBlur(blur_limit=(3, 7), p=0.3),
#             A.GaussNoise(var_limit=(10, 50), p=0.3),
#             A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#             ToTensorV2()
#         ])
        
#         # 敏感类增强变换（更保守）
#         self.sensitive_transform = A.Compose([
#             A.RandomRotate90(p=0.3),
#             A.HorizontalFlip(p=0.3),
#             A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
#             A.CoarseDropout(num_holes=2, max_height=3, max_width=3, p=0.3),
#             A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#             ToTensorV2()
#         ])
    
#     def get_transform(self, class_name):
#         """根据类别选择合适的增强策略"""
#         if class_name in self.sensitive_classes:
#             return self.sensitive_transform
#         return self.base_transform
    
#     def process_image(self, image_path, class_name):
#         """处理单张图片并生成增强版本"""
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
#         transforms = self.get_transform(class_name)
        
#         augmented_images = []
#         for i in range(self.augmentations_per_image):
#             augmented = transforms(image=image)["image"]
#             augmented_images.append(augmented)
        
#         return augmented_images
    
#     def process_directory(self, input_dir, output_dir):
#         """处理整个目录的图片"""
#         # 创建输出目录
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 获取所有类别
#         classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        
#         for class_name in tqdm(classes, desc="Processing classes"):
#             class_input_dir = os.path.join(input_dir, class_name)
#             class_output_dir = os.path.join(output_dir, class_name)
#             os.makedirs(class_output_dir, exist_ok=True)
            
#             # 获取该类别的所有图片
#             image_files = [f for f in os.listdir(class_input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
#             for img_file in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
#                 img_path = os.path.join(class_input_dir, img_file)
#                 img_name = os.path.splitext(img_file)[0]
                
#                 # 保存原始图片（如果需要）
#                 if self.save_original:
#                     original_img = Image.open(img_path)
#                     original_img.save(os.path.join(class_output_dir, f"{img_name}_original.png"))
                
#                 # 生成并保存增强图片
#                 augmented_images = self.process_image(img_path, class_name)
#                 for i, aug_img in enumerate(augmented_images):
#                     # 转换为PIL图片并保存
#                     if isinstance(aug_img, np.ndarray):
#                         aug_img_pil = Image.fromarray(aug_img)
#                     else:
#                         # 如果是tensor格式，转换为PIL
#                         aug_img_pil = Image.fromarray((aug_img.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8))
#                     aug_img_pil.save(os.path.join(class_output_dir, f"{img_name}_aug_{i}.png"))

# def augment_dataset(input_dir, output_dir, augmentations_per_image=3, seed=42):
#     """对外暴露的增强函数，用于处理数据集"""
#     augmenter = ImageAugmenter(
#         augmentations_per_image=augmentations_per_image,
#         seed=seed,
#         save_original=True
#     )
#     augmenter.process_directory(input_dir, output_dir)
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
# from PIL import Image
# from pathlib import Path
# import logging
# from tqdm import tqdm

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
# logger = logging.getLogger(__name__)

# # 设备设置（优先GPU→MPS→CPU）
# def get_device():
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")

# device = get_device()
# logger.info(f"使用设备: {device}")


# # ===================== 1. 基础工具函数（Mixup/CutMix）【核心修复】 =====================
# def mixup_data(x, y, alpha=0.2):
#     """Mixup：线性混合两张图像及标签（训练时实时增强）- 修复y=None的情况"""
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)  # 混合比例（0~1）
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)  # 随机打乱样本索引
#     mixed_x = lam * x + (1 - lam) * x[index, :]
    
#     # 核心修复：如果y是None（GAN训练无标签），直接返回None，不处理标签
#     if y is None:
#         return mixed_x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return mixed_x, y_a, y_b, lam

# def cutmix_data(x, y, alpha=0.2):
#     """CutMix：裁剪一张图像的局部粘贴到另一张（训练时实时增强）- 修复y=None的情况"""
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size, _, H, W = x.size()
    
#     # 随机生成裁剪区域
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
    
#     # 裁剪边界（避免超出图像）
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
    
#     # 混合图像
#     index = torch.randperm(batch_size).to(x.device)
#     x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))  # 实际混合比例
    
#     # 核心修复：如果y是None（GAN训练无标签），直接返回None，不处理标签
#     if y is None:
#         return x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return x, y_a, y_b, lam


# # ===================== 2. 数据加载器（支持传统/自动化增强）【无修改】 =====================
# class ImageDataset(Dataset):
#     """加载图像数据集，集成传统增强+自动化增强"""
#     def __init__(
#         self, 
#         root_dir, 
#         img_size=(32, 32), 
#         use_auto_aug=False,  # 是否启用AutoAugment
#         use_rand_aug=False, # 是否启用RandAugment
#         is_train=True       # 训练集/测试集（测试集仅做Resize+归一化）
#     ):
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.is_train = is_train
#         self.image_paths = [
#             p for p in Path(root_dir).glob('**/*') 
#             if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
#         ]
#         logger.info(f"数据集加载完成，共 {len(self.image_paths)} 张图像（训练模式：{is_train}）")

#         # 构建增强Pipeline（分训练集/测试集）
#         self.transform = self._build_transform(use_auto_aug, use_rand_aug)

#     def _build_transform(self, use_auto_aug, use_rand_aug):
#         """构建增强流水线（整合网页12种传统技术+自动化增强）"""
#         transform_list = []

#         if self.is_train:
#             # 1. 几何变换（网页核心技术：RandomResizedCrop、Flipping、Rotation、Padding、Affine）
#             transform_list.extend([
#                 # RandomResizedCrop（裁剪+Resize，替代固定裁剪）
#                 transforms.RandomResizedCrop(
#                     size=self.img_size,
#                     scale=(0.08, 1.0),  # 裁剪区域占原图8%~100%
#                     ratio=(3/4, 4/3)    # 宽高比范围
#                 ),
#                 # 水平翻转
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 # 垂直翻转（补充网页未提及的常用技术）
#                 transforms.RandomVerticalFlip(p=0.2),
#                 # 随机旋转（±15°）
#                 transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#                 # 随机仿射（融合平移、缩放、剪切，替代单独Affine）
#                 transforms.RandomAffine(
#                     degrees=5,
#                     translate=(0.05, 0.05),  # 平移±5%
#                     scale=(0.95, 1.05),      # 缩放±5%
#                     shear=(5, 5),            # 剪切±5°
#                     fill=(255, 255, 255)
#                 ),
#                 # 边缘填充（可选，根据需求开启）
#                 # transforms.Pad(padding=10, fill=(255, 255, 255), padding_mode="constant")
#             ])

#             # 2. 自动化增强（二选一，避免重复）
#             if use_auto_aug:
#                 transform_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
#             if use_rand_aug and not use_auto_aug:
#                 transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

#             # 3. 像素/颜色变换（网页核心技术：GaussianBlur、Grayscale、ColorJitter）
#             transform_list.extend([
#                 # 高斯模糊（概率0.2）
#                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=0.2),
#                 # 灰度转换（概率0.1，输出3通道兼容RGB模型）
#                 transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#                 # 颜色抖动（亮度/对比度/饱和度/色调）
#                 transforms.ColorJitter(
#                     brightness=0.2,
#                     contrast=0.2,
#                     saturation=0.2,
#                     hue=0.1
#                 )
#             ])

#         # 4. 固定预处理（所有模式通用：Resize+归一化+转Tensor）
#         transform_list.extend([
#             transforms.Resize(self.img_size),  # 确保最终尺寸统一
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化到[-1,1]
#         ])

#         return transforms.Compose(transform_list)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             image = Image.open(img_path).convert('RGB')
#             return self.transform(image)  # 仅返回图像（无标签，GAN训练不需要）
#         except Exception as e:
#             logger.warning(f"跳过损坏图像 {img_path}：{e}")
#             # 返回随机张量避免训练中断（实际需确保数据集无损坏）
#             return torch.randn(3, self.img_size[0], self.img_size[1])


# # ===================== 3. GAN模型定义（生成器+判别器，保留自注意力）【无修改】 =====================
# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, C, w, h = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, w*h).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, w*h)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(batch_size, -1, w*h)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, w, h)
#         return self.gamma * out + x


# class Generator(nn.Module):
#     def __init__(self, latent_dim=100, channels=3, img_size=32):
#         super().__init__()
#         self.init_size = img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size **2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             SelfAttention(64),
            
#             nn.Conv2d(64, channels, 3, 1, 1),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         return self.conv_blocks(out)


# class Discriminator(nn.Module):
#     def __init__(self, channels=3, img_size=32):
#         super().__init__()
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         ds_size = img_size // (2**4)
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         return self.adv_layer(out)


# # ===================== 4. GAN训练器（支持Mixup/CutMix融入训练）【无修改】 =====================
# class GANTrainer:
#     def __init__(
#         self,
#         data_dir,
#         latent_dim=100,
#         img_size=(32, 32),
#         epochs=30,
#         batch_size=64,
#         lr=0.0002,
#         weight_path="generator_weights.pth",
#         use_mixup=False,    # 是否启用Mixup
#         use_cutmix=False    # 是否启用CutMix
#     ):
#         self.data_dir = data_dir
#         self.latent_dim = latent_dim
#         self.img_size = img_size
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_path = weight_path
#         self.use_mixup = use_mixup
#         self.use_cutmix = use_cutmix

#         # 初始化模型
#         self.generator = Generator(latent_dim=latent_dim, img_size=img_size[0]).to(device)
#         self.discriminator = Discriminator(img_size=img_size[0]).to(device)

#         # 损失与优化器
#         self.adversarial_loss = nn.BCELoss().to(device)
#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
#         self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

#         # 加载数据集（启用自动化增强）
#         self.dataset = ImageDataset(
#             root_dir=data_dir,
#             img_size=img_size,
#             use_rand_aug=True,  # 启用RandAugment提升GAN训练数据多样性
#             is_train=True
#         )
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True  # 加速GPU数据传输
#         )

#     def train(self):
#         logger.info(f"开始GAN训练（Mixup: {self.use_mixup}, CutMix: {self.use_cutmix}），共 {self.epochs} 轮")
        
#         for epoch in range(self.epochs):
#             pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
#             for imgs in pbar:
#                 batch_size = imgs.size(0)
#                 valid = torch.ones(batch_size, 1).to(device)
#                 fake = torch.zeros(batch_size, 1).to(device)
#                 real_imgs = imgs.to(device)

#                 # 可选：对真实图像应用Mixup/CutMix（提升GAN训练稳定性）
#                 if self.use_mixup:
#                     # 传入y=None（GAN无标签），修复后函数会跳过标签处理
#                     real_imgs, _, _, _ = mixup_data(real_imgs, None, alpha=0.1)
#                 if self.use_cutmix and not self.use_mixup:
#                     # 传入y=None（GAN无标签），修复后函数会跳过标签处理
#                     real_imgs, _, _, _ = cutmix_data(real_imgs, None, alpha=0.1)

#                 # ----------------- 训练生成器 -----------------
#                 self.optimizer_G.zero_grad()
#                 z = torch.randn(batch_size, self.latent_dim).to(device)
#                 gen_imgs = self.generator(z)
#                 g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
#                 g_loss.backward()
#                 self.optimizer_G.step()

#                 # ----------------- 训练判别器 -----------------
#                 self.optimizer_D.zero_grad()
#                 real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
#                 fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 d_loss = (real_loss + fake_loss) / 2
#                 d_loss.backward()
#                 self.optimizer_D.step()

#                 # 显示进度
#                 pbar.set_postfix({"D损失": d_loss.item(), "G损失": g_loss.item()})

#             # 每10轮保存中间权重
#             if (epoch + 1) % 10 == 0:
#                 torch.save(self.generator.state_dict(), f"generator_weights_epoch_{epoch+1}.pth")
#                 logger.info(f"已保存第 {epoch+1} 轮GAN权重")

#         torch.save(self.generator.state_dict(), self.weight_path)
#         logger.info(f"GAN训练完成，最终权重保存至 {self.weight_path}")


# # ===================== 5. 完整增强器（传统+自动化+GAN）【无修改】 =====================
# class FullAugmenter:
#     """整合所有增强技术的统一接口：传统增强+自动化增强+GAN增强"""
#     def __init__(
#         self,
#         img_size=(32, 32),
#         use_auto_aug=False,
#         use_rand_aug=True,
#         gan_weight_path="generator_weights.pth",
#         use_gan=True
#     ):
#         self.img_size = img_size
#         self.use_gan = use_gan

#         # 1. 传统+自动化增强Pipeline（用于生成真实变体）
#         self.traditional_transform = self._build_traditional_transform(use_auto_aug, use_rand_aug)
#         # 反归一化：将Tensor转回PIL图像
#         self.inv_transform = transforms.Compose([
#             transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
#             transforms.ToPILImage()
#         ])

#         # 2. GAN增强初始化（可选）
#         self.gan_available = False
#         if use_gan:
#             self.generator = Generator(latent_dim=100, img_size=img_size[0]).to(device)
#             self.gan_available = self._load_gan_weights(gan_weight_path)
#             self.gan_preprocess = transforms.Compose([
#                 transforms.Resize(img_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#             ])

#     def _build_traditional_transform(self, use_auto_aug, use_rand_aug):
#         """构建传统+自动化增强流水线（对应网页12种技术）"""
#         transform_list = [
#             # 几何变换
#             transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#             transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#             # 自动化增强
#             transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#             # 像素变换
#             transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
#             transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             # 固定预处理
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ]
#         return transforms.Compose(transform_list)

#     def _load_gan_weights(self, weight_path):
#         """加载GAN生成器权重"""
#         if os.path.exists(weight_path):
#             try:
#                 self.generator.load_state_dict(torch.load(weight_path, map_location=device))
#                 self.generator.eval()
#                 logger.info(f"成功加载GAN权重：{weight_path}")
#                 return True
#             except Exception as e:
#                 logger.error(f"GAN权重加载失败：{e}")
#                 return False
#         else:
#             logger.warning(f"未找到GAN权重文件：{weight_path}")
#             return False

#     def traditional_augment(self, image: Image.Image) -> Image.Image:
#         """生成传统+自动化增强的样本"""
#         img_tensor = self.traditional_transform(image)
#         return self.inv_transform(img_tensor)

#     def gan_augment(self, image: Image.Image) -> Image.Image | None:
#         """生成GAN增强的样本（融合原始图像特征）"""
#         if not self.gan_available:
#             return None
#         with torch.no_grad():
#             img_tensor = self.gan_preprocess(image).unsqueeze(0).to(device)
#             z = torch.randn(1, 100, device=device)
#             gen_img = self.generator(z)
#             fused_img = 0.6 * gen_img + 0.4 * img_tensor  # 加权融合，保留原始特征
#             return self.inv_transform(fused_img.squeeze(0).cpu())

#     def augment(self, image: Image.Image, use_gan: bool = True) -> list[Image.Image]:
#         """统一增强接口：返回传统增强+GAN增强的样本列表"""
#         aug_imgs = [self.traditional_augment(image)]  # 至少1个传统增强样本
#         if use_gan and self.gan_available:
#             gan_img = self.gan_augment(image)
#             if gan_img:
#                 aug_imgs.append(gan_img)
#         return aug_imgs


# # ===================== 6. 主流程：GAN训练+全量数据增强【无修改】 =====================
# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 10,
#     img_size: tuple = (32, 32),
#     train_gan: bool = True,
#     gan_epochs: int = 100,
#     use_auto_aug: bool = True,
#     use_rand_aug: bool = True,
#     use_gan: bool = True,
#     use_mixup_in_gan: bool = True
# ):
#     """
#     完整数据增强流程：
#     1. 训练GAN（可选）
#     2. 用传统+自动化+GAN增强生成样本
#     3. 保留原始目录结构保存结果
#     """
#     weight_path = "generator_weights.pth"

#     # 步骤1：训练GAN（若需要）
#     if train_gan or not os.path.exists(weight_path):
#         trainer = GANTrainer(
#             data_dir=input_dir,
#             img_size=img_size,
#             epochs=gan_epochs,
#             use_mixup=use_mixup_in_gan,
#             weight_path=weight_path
#         )
#         trainer.train()
#     else:
#         logger.info("检测到已有GAN权重，跳过训练")

#     # 步骤2：初始化全量增强器
#     augmenter = FullAugmenter(
#         img_size=img_size,
#         use_auto_aug=use_auto_aug,
#         use_rand_aug=use_rand_aug,
#         gan_weight_path=weight_path,
#         use_gan=use_gan
#     )

#     # 步骤3：批量处理图像
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     # 查找所有图像文件
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
#     image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
#     logger.info(f"共找到 {len(image_files)} 张图像，开始全量增强...")

#     for img_path in image_files:
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             logger.warning(f"跳过无效图像 {img_path}：{e}")
#             continue

#         # 保持原始目录结构
#         rel_dir = img_path.parent.relative_to(input_path)
#         target_dir = output_path / rel_dir
#         target_dir.mkdir(parents=True, exist_ok=True)

#         # 保存原始图像
#         orig_path = target_dir / f"orig_{img_path.name}"
#         image.save(orig_path)

#         # 生成增强图像（按指定数量生成）
#         for i in range(augmentations_per_image):
#             # 随机选择增强组合（70%传统+30%GAN，无GAN则全用传统）
#             use_gan_flag = use_gan and augmenter.gan_available and random.random() < 1
#             aug_imgs = augmenter.augment(image, use_gan=use_gan_flag)
            
#             # 保存增强样本（确保数量达标）
#             for j, aug_img in enumerate(aug_imgs):
#                 if i * 2 + j >= augmentations_per_image:
#                     break  # 避免超出指定数量
#                 aug_save_path = target_dir / f"aug_{i}_{j}_{img_path.name}"
#                 aug_img.save(aug_save_path)

#     logger.info(f"全量数据增强完成！结果保存至 {output_dir}")

#74.58
# import os 
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
# from PIL import Image
# from pathlib import Path
# import logging
# from tqdm import tqdm

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
# logger = logging.getLogger(__name__)

# # 设备设置（优先GPU→MPS→CPU）
# def get_device():
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")

# device = get_device()
# logger.info(f"使用设备: {device}")


# # ===================== 1. 基础工具函数（Mixup/CutMix）【无修改】 =====================
# def mixup_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     if y is None:
#         return mixed_x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return mixed_x, y_a, y_b, lam

# def cutmix_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size, _, H, W = x.size()
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#     index = torch.randperm(batch_size).to(x.device)
#     x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
#     if y is None:
#         return x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return x, y_a, y_b, lam


# # ===================== 2. 数据加载器（支持传统/自动化增强）【无修改】 =====================
# class ImageDataset(Dataset):
#     def __init__(
#         self, 
#         root_dir, 
#         img_size=(32, 32), 
#         use_auto_aug=False,
#         use_rand_aug=False,
#         is_train=True
#     ):
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.is_train = is_train
#         self.image_paths = [
#             p for p in Path(root_dir).glob('**/*') 
#             if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
#         ]
#         logger.info(f"数据集加载完成，共 {len(self.image_paths)} 张图像（训练模式：{is_train}）")
#         self.transform = self._build_transform(use_auto_aug, use_rand_aug)

#     def _build_transform(self, use_auto_aug, use_rand_aug):
#         transform_list = []
#         if self.is_train:
#             transform_list.extend([
#                 transforms.RandomResizedCrop(size=self.img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.2),
#                 transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#                 transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#                 transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=0.2),
#                 transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
#             ])
#         transform_list.extend([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#         return transforms.Compose(transform_list)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             image = Image.open(img_path).convert('RGB')
#             return self.transform(image)
#         except Exception as e:
#             logger.warning(f"跳过损坏图像 {img_path}：{e}")
#             return torch.randn(3, self.img_size[0], self.img_size[1])


# # ===================== 3. GAN模型定义（彻底移除噪声相关）【核心修改】 =====================
# class SelfAttention(nn.Module):
#     """保留自注意力，无修改"""
#     def __init__(self, in_dim):
#         super().__init__()
#         self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, C, w, h = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, w*h).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, w*h)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(batch_size, -1, w*h)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, w, h)
#         return self.gamma * out + x


# class Generator(nn.Module):
#     """【修改1】移除latent_dim和噪声输入，改为固定维度的输入向量"""
#     def __init__(self, channels=3, img_size=32, input_dim=100):
#         # 用input_dim（固定输入维度）替代latent_dim，不再依赖外部噪声
#         super().__init__()
#         self.init_size = img_size // 4
#         self.input_dim = input_dim  # 固定输入维度（替代噪声维度）
#         # 【修改2】线性层输入维度改为input_dim（无噪声，仅用固定维度向量）
#         self.l1 = nn.Sequential(nn.Linear(self.input_dim, 128 * self.init_size ** 2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             SelfAttention(64),
            
#             nn.Conv2d(64, channels, 3, 1, 1),
#             nn.Tanh()
#         )

#     def forward(self):
#         """【修改3】无输入参数，内部生成固定维度的随机向量（替代外部噪声z）"""
#         # 内部生成随机向量（仅用于模型前向，无外部噪声依赖）
#         x = torch.randn(1, self.input_dim, device=device)  # 单样本生成
#         out = self.l1(x)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         return self.conv_blocks(out)


# class Discriminator(nn.Module):
#     """【无修改】判别器不涉及噪声，保持原结构"""
#     def __init__(self, channels=3, img_size=32):
#         super().__init__()
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         ds_size = img_size // (2**4)
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         return self.adv_layer(out)


# # ===================== 4. GAN训练器（彻底移除噪声相关）【核心修改】 =====================
# class GANTrainer:
#     def __init__(
#         self,
#         data_dir,
#         img_size=(32, 32),
#         epochs=30,
#         batch_size=64,
#         lr=0.0002,
#         weight_path="generator_weights.pth",
#         use_mixup=True,
#         use_cutmix=True,
#         generator_input_dim=100  # 【新增】生成器固定输入维度（替代latent_dim）
#     ):
#         self.data_dir = data_dir
#         self.img_size = img_size
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_path = weight_path
#         self.use_mixup = use_mixup
#         self.use_cutmix = use_cutmix
#         self.generator_input_dim = generator_input_dim

#         # 【修改1】初始化生成器：无latent_dim，传generator_input_dim
#         self.generator = Generator(
#             img_size=img_size[0],
#             input_dim=self.generator_input_dim
#         ).to(device)
#         self.discriminator = Discriminator(img_size=img_size[0]).to(device)

#         # 损失与优化器（无修改）
#         self.adversarial_loss = nn.BCELoss().to(device)
#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
#         self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

#         # 加载数据集（无修改）
#         self.dataset = ImageDataset(
#             root_dir=data_dir,
#             img_size=img_size,
#             use_rand_aug=True,
#             is_train=True
#         )
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True
#         )

#     def train(self):
#         logger.info(f"开始GAN训练（Mixup: {self.use_mixup}, CutMix: {self.use_cutmix}），共 {self.epochs} 轮")
        
#         for epoch in range(self.epochs):
#             pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
#             for imgs in pbar:
#                 batch_size = imgs.size(0)
#                 valid = torch.ones(batch_size, 1).to(device)
#                 fake = torch.zeros(batch_size, 1).to(device)
#                 real_imgs = imgs.to(device)

#                 # Mixup/CutMix处理（无修改）
#                 if self.use_mixup:
#                     real_imgs, _, _, _ = mixup_data(real_imgs, None, alpha=0.1)
#                 if self.use_cutmix and not self.use_mixup:
#                     real_imgs, _, _, _ = cutmix_data(real_imgs, None, alpha=0.1)

#                 # ----------------- 训练生成器【修改2】移除噪声z，直接调用generator() -----------------
#                 self.optimizer_G.zero_grad()
#                 # 无外部噪声，生成器内部生成输入向量
#                 gen_imgs = torch.cat([self.generator() for _ in range(batch_size)], dim=0)  # 批量生成
#                 g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
#                 g_loss.backward()
#                 self.optimizer_G.step()

#                 # ----------------- 训练判别器（无修改） -----------------
#                 self.optimizer_D.zero_grad()
#                 real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
#                 fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 d_loss = (real_loss + fake_loss) / 2
#                 d_loss.backward()
#                 self.optimizer_D.step()

#                 pbar.set_postfix({"D损失": d_loss.item(), "G损失": g_loss.item()})

#             # 保存权重（无修改）
#             if (epoch + 1) % 10 == 0:
#                 torch.save({
#                     "generator_state_dict": self.generator.state_dict(),
#                     "input_dim": self.generator_input_dim  # 保存输入维度，后续加载用
#                 }, f"generator_weights_epoch_{epoch+1}.pth")
#                 logger.info(f"已保存第 {epoch+1} 轮GAN权重")

#         torch.save({
#             "generator_state_dict": self.generator.state_dict(),
#             "input_dim": self.generator_input_dim
#         }, self.weight_path)
#         logger.info(f"GAN训练完成，最终权重保存至 {self.weight_path}")


# # ===================== 5. 完整增强器（彻底移除噪声相关）【核心修改】 =====================
# class FullAugmenter:
#     def __init__(
#         self,
#         img_size=(32, 32),
#         use_auto_aug=False,
#         use_rand_aug=True,
#         gan_weight_path="generator_weights.pth",
#         use_gan=True
#     ):
#         self.img_size = img_size
#         self.use_gan = use_gan

#         # 传统+自动化增强（无修改）
#         self.traditional_transform = self._build_traditional_transform(use_auto_aug, use_rand_aug)
#         self.inv_transform = transforms.Compose([
#             transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
#             transforms.ToPILImage()
#         ])

#         # 【修改1】GAN初始化：移除latent_dim，加载input_dim
#         self.gan_available = False
#         self.generator = None
#         self.generator_input_dim = 100  # 默认输入维度
#         if use_gan:
#             self.gan_available = self._load_gan_weights(gan_weight_path)
#             if self.gan_available:
#                 self.gan_preprocess = transforms.Compose([
#                     transforms.Resize(img_size),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#                 ])

#     def _build_traditional_transform(self, use_auto_aug, use_rand_aug):
#         """无修改，保留传统+自动化增强"""
#         transform_list = [
#             transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#             transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#             transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#             transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
#             transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ]
#         return transforms.Compose(transform_list)

#     def _load_gan_weights(self, weight_path):
#         """【修改2】加载权重时读取input_dim，无latent_dim"""
#         if not os.path.exists(weight_path):
#             logger.warning(f"未找到GAN权重文件：{weight_path}")
#             return False
#         try:
#             checkpoint = torch.load(weight_path, map_location=device)
#             self.generator_input_dim = checkpoint.get("input_dim", 100)
#             # 初始化生成器（无latent_dim，传input_dim）
#             self.generator = Generator(
#                 img_size=self.img_size[0],
#                 input_dim=self.generator_input_dim
#             ).to(device)
#             self.generator.load_state_dict(checkpoint["generator_state_dict"])
#             self.generator.eval()
#             logger.info(f"成功加载GAN权重：{weight_path}，输入维度：{self.generator_input_dim}")
#             return True
#         except Exception as e:
#             logger.error(f"GAN权重加载失败：{e}")
#             return False

#     def traditional_augment(self, image: Image.Image) -> Image.Image:
#         """无修改，传统+自动化增强"""
#         img_tensor = self.traditional_transform(image)
#         return self.inv_transform(img_tensor)

#     def gan_augment(self, image: Image.Image) -> Image.Image | None:
#         """【修改3】移除噪声z，直接调用generator()生成样本"""
#         if not self.gan_available or self.generator is None:
#             return None
#         with torch.no_grad():
#             # 无外部噪声，生成器内部生成样本
#             gen_img = self.generator()  # 单样本生成
#             # 融合原始图像特征（保持原逻辑）
#             img_tensor = self.gan_preprocess(image).unsqueeze(0).to(device)
#             fused_img = 0.6 * gen_img + 0.4 * img_tensor
#             return self.inv_transform(fused_img.squeeze(0).cpu())

#     def augment(self, image: Image.Image, use_gan: bool = True) -> list[Image.Image]:
#         """无修改，统一增强接口"""
#         aug_imgs = [self.traditional_augment(image)]
#         if use_gan and self.gan_available:
#             gan_img = self.gan_augment(image)
#             if gan_img:
#                 aug_imgs.append(gan_img)
#         return aug_imgs


# # ===================== 6. 主流程：GAN训练+全量数据增强【修改：移除latent_dim参数】 =====================
# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 10,
#     img_size: tuple = (32, 32),
#     train_gan: bool = True,
#     gan_epochs: int = 30,
#     use_auto_aug: bool = True,
#     use_rand_aug: bool = True,
#     use_gan: bool = True,
#     use_mixup_in_gan: bool = True,
#     generator_input_dim=100  # 【新增】生成器输入维度（替代latent_dim）
# ):
#     weight_path = "generator_weights.pth"

#     # 步骤1：训练GAN（无latent_dim，传generator_input_dim）
#     if train_gan or not os.path.exists(weight_path):
#         trainer = GANTrainer(
#             data_dir=input_dir,
#             img_size=img_size,
#             epochs=gan_epochs,
#             use_mixup=use_mixup_in_gan,
#             weight_path=weight_path,
#             generator_input_dim=generator_input_dim
#         )
#         trainer.train()
#     else:
#         logger.info("检测到已有GAN权重，跳过训练")

#     # 步骤2：初始化增强器（无修改）
#     augmenter = FullAugmenter(
#         img_size=img_size,
#         use_auto_aug=use_auto_aug,
#         use_rand_aug=use_rand_aug,
#         gan_weight_path=weight_path,
#         use_gan=use_gan
#     )

#     # 步骤3：批量处理图像（无修改）
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
#     image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
#     logger.info(f"共找到 {len(image_files)} 张图像，开始全量增强...")

#     for img_path in image_files:
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             logger.warning(f"跳过无效图像 {img_path}：{e}")
#             continue

#         rel_dir = img_path.parent.relative_to(input_path)
#         target_dir = output_path / rel_dir
#         target_dir.mkdir(parents=True, exist_ok=True)

#         orig_path = target_dir / f"orig_{img_path.name}"
#         image.save(orig_path)

#         for i in range(augmentations_per_image):
#             use_gan_flag = use_gan and augmenter.gan_available and random.random() < 1
#             aug_imgs = augmenter.augment(image, use_gan=use_gan_flag)
            
#             for j, aug_img in enumerate(aug_imgs):
#                 if i * 2 + j >= augmentations_per_image:
#                     break
#                 aug_save_path = target_dir / f"aug_{i}_{j}_{img_path.name}"
#                 aug_img.save(aug_save_path)

#     logger.info(f"全量数据增强完成！结果保存至 {output_dir}")

#styleGAN 0.69
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import cv2
# from PIL import Image
# from pathlib import Path
# import logging
# from tqdm import tqdm

# # 配置日志（仅输出关键信息，避免冗余）
# logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-INFO-%(message)s')
# logger = logging.getLogger(__name__)

# # 设备设置（优先GPU→MPS→CPU）
# def get_device():
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")

# device = get_device()
# logger.info(f"使用设备: {device}")


# # ===================== 1. 基础工具函数（Mixup/CutMix）【无修改】 =====================
# def mixup_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     if y is None:
#         return mixed_x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return mixed_x, y_a, y_b, lam

# def cutmix_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size, _, H, W = x.size()
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#     index = torch.randperm(batch_size).to(x.device)
#     x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
#     if y is None:
#         return x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return x, y_a, y_b, lam


# # ===================== 2. 数据加载器（32×32）【无修改，仅保留必要日志】 =====================
# class ImageDataset(Dataset):
#     def __init__(
#         self, 
#         root_dir, 
#         img_size=(32, 32),
#         use_auto_aug=False,
#         use_rand_aug=False,
#         is_train=True
#     ):
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.is_train = is_train
#         self.image_paths = [
#             p for p in Path(root_dir).glob('**/*') 
#             if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
#         ]
#         logger.info(f"数据集加载完成：共 {len(self.image_paths)} 张图像，尺寸 {img_size}，训练模式 {is_train}")
#         self.transform = self._build_transform(use_auto_aug, use_rand_aug)

#     def _build_transform(self, use_auto_aug, use_rand_aug):
#         transform_list = []
#         if self.is_train:
#             transform_list.extend([
#                 transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#                 transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#                 transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=0.2),
#                 transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
#             ])
#         transform_list.extend([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#         return transforms.Compose(transform_list)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             image = Image.open(img_path).convert('RGB')
#             return self.transform(image)
#         except Exception as e:
#             logger.warning(f"跳过损坏图像：{img_path}，原因：{str(e)}")
#             return torch.randn(3, self.img_size[0], self.img_size[1])


# # ===================== 3. StyleGAN核心模块（删除所有调试打印） =====================
# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, C, w, h = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, w*h).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, w*h)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(batch_size, -1, w*h)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, w, h)
#         return self.gamma * out + x


# class AdaIN(nn.Module):
#     def __init__(self, in_channels, style_dim):
#         super().__init__()
#         self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False)
#         self.style_scale = nn.Linear(style_dim, in_channels)
#         self.style_shift = nn.Linear(style_dim, in_channels)
#         self.style_scale.weight.data.uniform_()
#         self.style_scale.bias.data.fill_(1.0)
#         self.style_shift.bias.data.fill_(0.0)

#     def forward(self, x, style_w):
#         x_norm = self.instance_norm(x)
#         scale = self.style_scale(style_w).view(-1, x_norm.size(1), 1, 1)
#         shift = self.style_shift(style_w).view(-1, x_norm.size(1), 1, 1)
#         return x_norm * scale + shift


# class StyleMappingNetwork(nn.Module):
#     def __init__(self, z_dim=512, w_dim=512, num_layers=8):
#         super().__init__()
#         self.z_dim = z_dim
#         self.w_dim = w_dim
#         layers = []
#         for _ in range(num_layers):
#             layers.append(nn.Linear(w_dim, w_dim))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#         self.mapping = nn.Sequential(*layers)

#     def forward(self, batch_size):
#         z = torch.randn(batch_size, self.z_dim, device=device)
#         w = self.mapping(z)
#         return w


# class StyleGANGenerator(nn.Module):
#     """32×32生成器（无任何调试打印）"""
#     def __init__(self, w_dim=512, img_size=32, channels=3):
#         super().__init__()
#         self.w_dim = w_dim
#         self.img_size = img_size
#         self.num_up_layers = 3  # 4→8→16→32（3次上采样）

#         # 初始卷积：1×1→4×4（256通道）
#         self.init_conv = nn.Conv2d(1, 256, kernel_size=4, padding=3)
#         self.init_norm = nn.InstanceNorm2d(256, affine=True)
#         self.init_act = nn.LeakyReLU(0.2, inplace=True)

#         # 上采样层（固定通道和尺寸变化）
#         self.synthesis_layers = nn.ModuleList([
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(256, 128, 3, padding=1),
#             AdaIN(128, w_dim),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(128, 64, 3, padding=1),
#             AdaIN(64, w_dim),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             AdaIN(64, w_dim),
#             nn.LeakyReLU(0.2, inplace=True),
#         ])

#         # 自注意力+输出卷积
#         self.attention = SelfAttention(64)
#         self.final_conv = nn.Conv2d(64, channels, 1, padding=0)
#         self.tanh = nn.Tanh()

#         # 映射网络
#         self.mapping_net = StyleMappingNetwork(z_dim=w_dim, w_dim=w_dim)

#     def forward(self, batch_size=1):
#         w = self.mapping_net(batch_size)
#         # 初始特征：1×1→4×4
#         x = torch.ones(batch_size, 1, 1, 1, device=device)
#         x = self.init_conv(x)
#         x = self.init_norm(x)
#         x = self.init_act(x)
#         # 上采样到32×32
#         for layer in self.synthesis_layers:
#             if isinstance(layer, AdaIN):
#                 x = layer(x, w)
#             else:
#                 x = layer(x)
#         # 自注意力+输出
#         x = self.attention(x)
#         gen_imgs = self.final_conv(x)
#         gen_imgs = self.tanh(gen_imgs)
#         return gen_imgs


# class StyleGANDiscriminator(nn.Module):
#     """32×32判别器（无任何调试打印）"""
#     def __init__(self, img_size=32, channels=3):
#         super().__init__()
#         self.img_size = img_size
#         # 固定4个尺度：32→16→8→4
#         self.discriminator_blocks = nn.ModuleList([
#             # 尺度1：32→16（3→64→128）
#             nn.Sequential(
#                 nn.Conv2d(channels, 64, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(64, 128, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.AvgPool2d(2, stride=2)
#             ),
#             # 尺度2：16→8（128→256）
#             nn.Sequential(
#                 nn.Conv2d(128, 256, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.AvgPool2d(2, stride=2)
#             ),
#             # 尺度3：8→4（256→512）
#             nn.Sequential(
#                 nn.Conv2d(256, 512, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(512, 512, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.AvgPool2d(2, stride=2)
#             ),
#             # 尺度4：4×4（512→512）
#             nn.Sequential(
#                 nn.Conv2d(512, 512, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(512, 512, 3, padding=1),
#                 nn.LeakyReLU(0.2, inplace=True)
#             )
#         ])
#         # 线性层：512×4×4=8192
#         self.final_linear = nn.Linear(512 * 4 * 4, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, imgs):
#         x = imgs
#         # 逐尺度处理
#         for block in self.discriminator_blocks:
#             x = block(x)
#         # 展平+线性层
#         x_flat = x.view(imgs.size(0), -1)
#         output = self.final_linear(x_flat)
#         output = self.sigmoid(output)
#         return output


# # ===================== 4. StyleGAN训练器（仅保留进度条和关键日志） =====================
# class StyleGANTrainer:
#     def __init__(
#         self,
#         data_dir,
#         img_size=(32, 32),
#         epochs=50,
#         batch_size=64,
#         lr=2e-4,
#         weight_path="stylegan_generator_weights_32x32.pth",
#         use_mixup=True,
#         use_cutmix=True,
#         w_dim=512
#     ):
#         self.data_dir = data_dir
#         self.img_size = img_size
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_path = weight_path
#         self.use_mixup = use_mixup
#         self.use_cutmix = use_cutmix
#         self.w_dim = w_dim

#         # 初始化模型
#         self.generator = StyleGANGenerator(w_dim=w_dim, img_size=img_size[0]).to(device)
#         self.discriminator = StyleGANDiscriminator(img_size=img_size[0]).to(device)

#         # 优化器
#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.0, 0.99), weight_decay=1e-8)
#         self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.0, 0.99), weight_decay=1e-8)

#         # 损失函数
#         self.adversarial_loss = nn.BCELoss().to(device)

#         # 加载数据集
#         self.dataset = ImageDataset(root_dir=data_dir, img_size=img_size, use_rand_aug=True, is_train=True)
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True,
#             multiprocessing_context='spawn'
#         )

#     def train(self):
#         logger.info(f"StyleGAN训练开始：尺寸 {self.img_size}，轮次 {self.epochs}，批量 {self.batch_size}，Mixup={self.use_mixup}，CutMix={self.use_cutmix}")
        
#         for epoch in range(self.epochs):
#             # 进度条（仅显示轮次、进度、损失）
#             pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")
#             for imgs in pbar:
#                 batch_size = imgs.size(0)
#                 valid = torch.ones(batch_size, 1).to(device)
#                 fake = torch.zeros(batch_size, 1).to(device)
#                 real_imgs = imgs.to(device)

#                 # Mixup/CutMix处理
#                 if self.use_mixup:
#                     real_imgs, _, _, _ = mixup_data(real_imgs, None, alpha=0.1)
#                 elif self.use_cutmix:
#                     real_imgs, _, _, _ = cutmix_data(real_imgs, None, alpha=0.1)

#                 # 训练生成器
#                 self.optimizer_G.zero_grad()
#                 gen_imgs = self.generator(batch_size=batch_size)
#                 g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
#                 g_loss.backward()
#                 self.optimizer_G.step()

#                 # 训练判别器
#                 self.optimizer_D.zero_grad()
#                 d_real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
#                 d_fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 d_loss = (d_real_loss + d_fake_loss) / 2
#                 d_loss.backward()
#                 self.optimizer_D.step()

#                 # 进度条更新损失（仅保留4位小数）
#                 pbar.set_postfix({"D_loss": round(d_loss.item(), 4), "G_loss": round(g_loss.item(), 4)})

#             # 每10轮保存权重
#             if (epoch + 1) % 10 == 0:
#                 save_path = f"stylegan_generator_weights_32x32_epoch_{epoch+1}.pth"
#                 torch.save({
#                     "generator_state_dict": self.generator.state_dict(),
#                     "w_dim": self.w_dim,
#                     "img_size": self.img_size[0]
#                 }, save_path)
#                 logger.info(f"第 {epoch+1} 轮权重保存完成：{save_path}")

#         # 训练结束保存最终权重
#         torch.save({
#             "generator_state_dict": self.generator.state_dict(),
#             "w_dim": self.w_dim,
#             "img_size": self.img_size[0]
#         }, self.weight_path)
#         logger.info(f"StyleGAN训练完成！最终权重保存至：{self.weight_path}")


# # ===================== 5. 完整增强器（简洁版，无冗余打印） =====================
# class FullAugmenter:
#     def __init__(
#         self,
#         img_size=(32, 32),
#         use_auto_aug=False,
#         use_rand_aug=True,
#         stylegan_weight_path="stylegan_generator_weights_32x32.pth",
#         use_gan=True
#     ):
#         self.img_size = img_size
#         self.use_gan = use_gan
#         self.stylegan_weight_path = stylegan_weight_path

#         # 初始化传统增强器
#         self.traditional_transform = self._build_traditional_transform(use_auto_aug, use_rand_aug)
#         self.inv_transform = transforms.Compose([
#             transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
#             transforms.ToPILImage()
#         ])

#         # 初始化StyleGAN增强器（仅当use_gan=True时）
#         self.generator = None
#         if self.use_gan:
#             self._load_stylegan_model()

#     def _build_traditional_transform(self, use_auto_aug, use_rand_aug):
#         transform_list = []
#         if use_rand_aug or use_auto_aug:
#             transform_list.extend([
#                 transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#                 transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#                 transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=0.2),
#                 transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
#             ])
#         transform_list.extend([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#         return transforms.Compose(transform_list)

#     def _load_stylegan_model(self):
#         """加载StyleGAN模型（仅打印关键日志）"""
#         if not os.path.exists(self.stylegan_weight_path):
#             logger.error(f"StyleGAN权重文件不存在：{self.stylegan_weight_path}，将禁用GAN增强")
#             self.use_gan = False
#             return

#         try:
#             checkpoint = torch.load(self.stylegan_weight_path, map_location=device)
#             img_size = checkpoint.get("img_size", 32)
#             w_dim = checkpoint.get("w_dim", 512)

#             if img_size != self.img_size[0]:
#                 logger.error(f"StyleGAN权重尺寸不匹配：权重{img_size}×{img_size}，需求{self.img_size[0]}×{self.img_size[0]}，禁用GAN增强")
#                 self.use_gan = False
#                 return

#             # 初始化生成器
#             self.generator = StyleGANGenerator(w_dim=w_dim, img_size=img_size).to(device)
#             self.generator.load_state_dict(checkpoint["generator_state_dict"])
#             self.generator.eval()  # 推理模式
#             logger.info(f"StyleGAN模型加载完成：权重{self.stylegan_weight_path}，尺寸{img_size}×{img_size}")
#         except Exception as e:
#             logger.error(f"StyleGAN模型加载失败：{str(e)}，禁用GAN增强")
#             self.use_gan = False

#     def traditional_augment(self, image: Image.Image) -> Image.Image:
#         """传统数据增强（返回32×32图像）"""
#         img_tensor = self.traditional_transform(image)
#         return self.inv_transform(img_tensor)

#     def stylegan_augment(self, image: Image.Image) -> Image.Image | None:
#         """StyleGAN增强（仅当模型加载成功时可用）"""
#         if not self.use_gan or self.generator is None:
#             return None

#         with torch.no_grad():  # 禁用梯度计算，加速推理
#             # 生成GAN样本（32×32）
#             gen_img = self.generator(batch_size=1)
#             # 融合真实图像特征（避免生成样本与原图差异过大）
#             real_tensor = self.traditional_transform(image).unsqueeze(0).to(device)
#             fused_img = 0.6 * gen_img + 0.4 * real_tensor  # 权重可调整
#             return self.inv_transform(fused_img.squeeze(0))

#     def augment(self, image: Image.Image) -> list[Image.Image]:
#         """单次增强：返回[传统增强图, GAN增强图]（GAN增强失败则仅返回传统增强图）"""
#         aug_imgs = [self.traditional_augment(image)]
#         if self.use_gan:
#             gan_img = self.stylegan_augment(image)
#             if gan_img is not None:
#                 aug_imgs.append(gan_img)
#         return aug_imgs


# # ===================== 6. 主流程（数据增强入口，简洁日志） =====================
# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 10,
#     img_size=(32, 32),
#     train_gan: bool = True,
#     gan_epochs: int = 30,
#     use_auto_aug: bool = False,
#     use_rand_aug: bool = True,
#     use_gan: bool = True,
#     use_mixup_in_gan: bool = True,
#     w_dim=512
# ):
#     # 输入输出目录检查
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     if not input_path.exists():
#         logger.error(f"输入目录不存在：{input_dir}")
#         raise FileNotFoundError(f"Input directory not found: {input_dir}")
#     output_path.mkdir(parents=True, exist_ok=True)
#     logger.info(f"数据增强开始：输入目录{input_dir}，输出目录{output_dir}，每张图增强{augmentations_per_image}次")

#     # 第一步：训练StyleGAN（如需）
#     weight_path = "stylegan_generator_weights_32x32.pth"
#     if train_gan and use_gan:
#         # 强制删除旧权重（避免版本冲突）
#         old_weights = [f for f in Path.cwd().glob("stylegan_generator_weights_32x32*.pth")]
#         if old_weights:
#             for f in old_weights:
#                 os.remove(f)
#                 logger.info(f"删除旧权重文件：{f.name}")
#         # 训练StyleGAN
#         trainer = StyleGANTrainer(
#             data_dir=input_dir,
#             img_size=img_size,
#             epochs=gan_epochs,
#             batch_size=64,
#             lr=2e-4,
#             weight_path=weight_path,
#             use_mixup=use_mixup_in_gan,
#             use_cutmix=use_mixup_in_gan,  # Mixup/CutMix二选一
#             w_dim=w_dim
#         )
#         trainer.train()
#     elif use_gan and not os.path.exists(weight_path):
#         logger.error(f"未开启StyleGAN训练且权重文件不存在：{weight_path}，将禁用GAN增强")
#         use_gan = False

#     # 第二步：加载增强器并执行增强
#     augmenter = FullAugmenter(
#         img_size=img_size,
#         use_auto_aug=use_auto_aug,
#         use_rand_aug=use_rand_aug,
#         stylegan_weight_path=weight_path if use_gan else "",
#         use_gan=use_gan
#     )

#     # 获取所有图像文件（仅处理常见图像格式）
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
#     image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
#     total_images = len(image_files)
#     logger.info(f"共发现{total_images}张图像，开始批量增强...")

#     # 逐图增强并保存
#     for img_idx, img_file in enumerate(image_files, 1):
#         # 读取图像
#         try:
#             image = Image.open(img_file).convert('RGB')
#         except Exception as e:
#             logger.warning(f"跳过损坏图像（{img_idx}/{total_images}）：{img_file.name}，原因：{str(e)}")
#             continue

#         # 保持原始目录结构
#         relative_path = img_file.relative_to(input_path)
#         save_dir = output_path / relative_path.parent
#         save_dir.mkdir(parents=True, exist_ok=True)

#         # 单次图像增强augmentations_per_image次
#         for aug_idx in range(augmentations_per_image):
#             # 获取增强后的图像列表（1张传统增强图，可选1张GAN增强图）
#             aug_imgs = augmenter.augment(image)
#             # 保存增强图（命名规则：原图名_增强次数_类型.jpg）
#             for img_type_idx, aug_img in enumerate(aug_imgs):
#                 img_type = "traditional" if img_type_idx == 0 else "gan"
#                 save_name = f"{img_file.stem}_aug{aug_idx+1}_{img_type}.jpg"
#                 save_path = save_dir / save_name
#                 aug_img.save(save_path, quality=95)  # 保存为JPG，质量95

#         # 每处理10%的图像打印进度
#         if (img_idx % max(1, total_images // 10)) == 0:
#             progress = (img_idx / total_images) * 100
#             logger.info(f"数据增强进度：{img_idx}/{total_images}张（{progress:.1f}%）")

#     # 增强完成
#     logger.info(f"数据增强全部完成！增强后图像保存至：{output_dir}")

# #74.58 最好的Augmentation
# import os 
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
# from PIL import Image
# from pathlib import Path
# import logging
# from tqdm import tqdm

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
# logger = logging.getLogger(__name__)

# # 设备设置（优先GPU→MPS→CPU）
# def get_device():
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")

# device = get_device()
# logger.info(f"使用设备: {device}")


# # ===================== 1. 基础工具函数（Mixup/CutMix）【无修改】 =====================
# def mixup_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     if y is None:
#         return mixed_x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return mixed_x, y_a, y_b, lam

# def cutmix_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size, _, H, W = x.size()
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#     index = torch.randperm(batch_size).to(x.device)
#     x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
#     if y is None:
#         return x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return x, y_a, y_b, lam


# # ===================== 2. 数据加载器（支持传统/自动化增强）【无修改】 =====================
# class ImageDataset(Dataset):
#     def __init__(
#         self, 
#         root_dir, 
#         img_size=(32, 32), 
#         use_auto_aug=False,
#         use_rand_aug=False,
#         is_train=True
#     ):
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.is_train = is_train
#         self.image_paths = [
#             p for p in Path(root_dir).glob('**/*') 
#             if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
#         ]
#         logger.info(f"数据集加载完成，共 {len(self.image_paths)} 张图像（训练模式：{is_train}）")
#         self.transform = self._build_transform(use_auto_aug, use_rand_aug)

#     def _build_transform(self, use_auto_aug, use_rand_aug):
#         transform_list = []
#         if self.is_train:
#             transform_list.extend([
#                 transforms.RandomResizedCrop(size=self.img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.2),
#                 transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#                 transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#                 transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=0.2),
#                 transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
#             ])
#         transform_list.extend([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#         return transforms.Compose(transform_list)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             image = Image.open(img_path).convert('RGB')
#             return self.transform(image)
#         except Exception as e:
#             logger.warning(f"跳过损坏图像 {img_path}：{e}")
#             return torch.randn(3, self.img_size[0], self.img_size[1])


# # ===================== 3. GAN模型定义（彻底移除噪声相关）【核心修改】 =====================
# class SelfAttention(nn.Module):
#     """保留自注意力，无修改"""
#     def __init__(self, in_dim):
#         super().__init__()
#         self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, C, w, h = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, w*h).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, w*h)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(batch_size, -1, w*h)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, w, h)
#         return self.gamma * out + x


# class Generator(nn.Module):
#     """【修改1】移除latent_dim和噪声输入，改为固定维度的输入向量"""
#     def __init__(self, channels=3, img_size=32, input_dim=100):
#         # 用input_dim（固定输入维度）替代latent_dim，不再依赖外部噪声
#         super().__init__()
#         self.init_size = img_size // 4
#         self.input_dim = input_dim  # 固定输入维度（替代噪声维度）
#         # 【修改2】线性层输入维度改为input_dim（无噪声，仅用固定维度向量）
#         self.l1 = nn.Sequential(nn.Linear(self.input_dim, 128 * self.init_size ** 2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             SelfAttention(64),
            
#             nn.Conv2d(64, channels, 3, 1, 1),
#             nn.Tanh()
#         )

#     def forward(self):
#         """【修改3】无输入参数，内部生成固定维度的随机向量（替代外部噪声z）"""
#         # 内部生成随机向量（仅用于模型前向，无外部噪声依赖）
#         x = torch.randn(1, self.input_dim, device=device)  # 单样本生成
#         out = self.l1(x)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         return self.conv_blocks(out)


# class Discriminator(nn.Module):
#     """【无修改】判别器不涉及噪声，保持原结构"""
#     def __init__(self, channels=3, img_size=32):
#         super().__init__()
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         ds_size = img_size // (2**4)
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         return self.adv_layer(out)


# # ===================== 4. GAN训练器（彻底移除噪声相关）【核心修改】 =====================
# class GANTrainer:
#     def __init__(
#         self,
#         data_dir,
#         img_size=(32, 32),
#         epochs=30,
#         batch_size=64,
#         lr=0.0002,
#         weight_path="generator_weights.pth",
#         use_mixup=True,
#         use_cutmix=True,
#         generator_input_dim=100  # 【新增】生成器固定输入维度（替代latent_dim）
#     ):
#         self.data_dir = data_dir
#         self.img_size = img_size
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_path = weight_path
#         self.use_mixup = use_mixup
#         self.use_cutmix = use_cutmix
#         self.generator_input_dim = generator_input_dim

#         # 【修改1】初始化生成器：无latent_dim，传generator_input_dim
#         self.generator = Generator(
#             img_size=img_size[0],
#             input_dim=self.generator_input_dim
#         ).to(device)
#         self.discriminator = Discriminator(img_size=img_size[0]).to(device)

#         # 损失与优化器（无修改）
#         self.adversarial_loss = nn.BCELoss().to(device)
#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
#         self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

#         # 加载数据集（无修改）
#         self.dataset = ImageDataset(
#             root_dir=data_dir,
#             img_size=img_size,
#             use_rand_aug=True,
#             is_train=True
#         )
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True
#         )

#     def train(self):
#         logger.info(f"开始GAN训练（Mixup: {self.use_mixup}, CutMix: {self.use_cutmix}），共 {self.epochs} 轮")
        
#         for epoch in range(self.epochs):
#             pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
#             for imgs in pbar:
#                 batch_size = imgs.size(0)
#                 valid = torch.ones(batch_size, 1).to(device)
#                 fake = torch.zeros(batch_size, 1).to(device)
#                 real_imgs = imgs.to(device)

#                 # Mixup/CutMix处理（无修改）
#                 if self.use_mixup:
#                     real_imgs, _, _, _ = mixup_data(real_imgs, None, alpha=0.1)
#                 if self.use_cutmix and not self.use_mixup:
#                     real_imgs, _, _, _ = cutmix_data(real_imgs, None, alpha=0.1)

#                 # ----------------- 训练生成器【修改2】移除噪声z，直接调用generator() -----------------
#                 self.optimizer_G.zero_grad()
#                 # 无外部噪声，生成器内部生成输入向量
#                 gen_imgs = torch.cat([self.generator() for _ in range(batch_size)], dim=0)  # 批量生成
#                 g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
#                 g_loss.backward()
#                 self.optimizer_G.step()

#                 # ----------------- 训练判别器（无修改） -----------------
#                 self.optimizer_D.zero_grad()
#                 real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
#                 fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 d_loss = (real_loss + fake_loss) / 2
#                 d_loss.backward()
#                 self.optimizer_D.step()

#                 pbar.set_postfix({"D损失": d_loss.item(), "G损失": g_loss.item()})

#             # 保存权重（无修改）
#             if (epoch + 1) % 10 == 0:
#                 torch.save({
#                     "generator_state_dict": self.generator.state_dict(),
#                     "input_dim": self.generator_input_dim  # 保存输入维度，后续加载用
#                 }, f"generator_weights_epoch_{epoch+1}.pth")
#                 logger.info(f"已保存第 {epoch+1} 轮GAN权重")

#         torch.save({
#             "generator_state_dict": self.generator.state_dict(),
#             "input_dim": self.generator_input_dim
#         }, self.weight_path)
#         logger.info(f"GAN训练完成，最终权重保存至 {self.weight_path}")


# # ===================== 5. 完整增强器（彻底移除噪声相关）【核心修改】 =====================
# class FullAugmenter:
#     def __init__(
#         self,
#         img_size=(32, 32),
#         use_auto_aug=False,
#         use_rand_aug=True,
#         gan_weight_path="generator_weights.pth",
#         use_gan=True
#     ):
#         self.img_size = img_size
#         self.use_gan = use_gan

#         # 传统+自动化增强（无修改）
#         self.traditional_transform = self._build_traditional_transform(use_auto_aug, use_rand_aug)
#         self.inv_transform = transforms.Compose([
#             transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
#             transforms.ToPILImage()
#         ])

#         # 【修改1】GAN初始化：移除latent_dim，加载input_dim
#         self.gan_available = False
#         self.generator = None
#         self.generator_input_dim = 100  # 默认输入维度
#         if use_gan:
#             self.gan_available = self._load_gan_weights(gan_weight_path)
#             if self.gan_available:
#                 self.gan_preprocess = transforms.Compose([
#                     transforms.Resize(img_size),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#                 ])

#     def _build_traditional_transform(self, use_auto_aug, use_rand_aug):
#         """无修改，保留传统+自动化增强"""
#         transform_list = [
#             transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#             transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#             transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#             transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
#             transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ]
#         return transforms.Compose(transform_list)

#     def _load_gan_weights(self, weight_path):
#         """【修改2】加载权重时读取input_dim，无latent_dim"""
#         if not os.path.exists(weight_path):
#             logger.warning(f"未找到GAN权重文件：{weight_path}")
#             return False
#         try:
#             checkpoint = torch.load(weight_path, map_location=device)
#             self.generator_input_dim = checkpoint.get("input_dim", 100)
#             # 初始化生成器（无latent_dim，传input_dim）
#             self.generator = Generator(
#                 img_size=self.img_size[0],
#                 input_dim=self.generator_input_dim
#             ).to(device)
#             self.generator.load_state_dict(checkpoint["generator_state_dict"])
#             self.generator.eval()
#             logger.info(f"成功加载GAN权重：{weight_path}，输入维度：{self.generator_input_dim}")
#             return True
#         except Exception as e:
#             logger.error(f"GAN权重加载失败：{e}")
#             return False

#     def traditional_augment(self, image: Image.Image) -> Image.Image:
#         """无修改，传统+自动化增强"""
#         img_tensor = self.traditional_transform(image)
#         return self.inv_transform(img_tensor)

#     def gan_augment(self, image: Image.Image) -> Image.Image | None:
#         """【修改3】移除噪声z，直接调用generator()生成样本"""
#         if not self.gan_available or self.generator is None:
#             return None
#         with torch.no_grad():
#             # 无外部噪声，生成器内部生成样本
#             gen_img = self.generator()  # 单样本生成
#             # 融合原始图像特征（保持原逻辑）
#             img_tensor = self.gan_preprocess(image).unsqueeze(0).to(device)
#             fused_img = 0.6 * gen_img + 0.4 * img_tensor
#             return self.inv_transform(fused_img.squeeze(0).cpu())

#     def augment(self, image: Image.Image, use_gan: bool = True) -> list[Image.Image]:
#         """无修改，统一增强接口"""
#         aug_imgs = [self.traditional_augment(image)]
#         if use_gan and self.gan_available:
#             gan_img = self.gan_augment(image)
#             if gan_img:
#                 aug_imgs.append(gan_img)
#         return aug_imgs


# # ===================== 6. 主流程：GAN训练+全量数据增强【修改：移除latent_dim参数】 =====================
# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 10,
#     img_size: tuple = (32, 32),
#     train_gan: bool = True,
#     gan_epochs: int = 30,
#     use_auto_aug: bool = True,
#     use_rand_aug: bool = True,
#     use_gan: bool = True,
#     use_mixup_in_gan: bool = True,
#     generator_input_dim=100  # 【新增】生成器输入维度（替代latent_dim）
# ):
#     weight_path = "generator_weights.pth"

#     # 步骤1：训练GAN（无latent_dim，传generator_input_dim）
#     if train_gan or not os.path.exists(weight_path):
#         trainer = GANTrainer(
#             data_dir=input_dir,
#             img_size=img_size,
#             epochs=gan_epochs,
#             use_mixup=use_mixup_in_gan,
#             weight_path=weight_path,
#             generator_input_dim=generator_input_dim
#         )
#         trainer.train()
#     else:
#         logger.info("检测到已有GAN权重，跳过训练")

#     # 步骤2：初始化增强器（无修改）
#     augmenter = FullAugmenter(
#         img_size=img_size,
#         use_auto_aug=use_auto_aug,
#         use_rand_aug=use_rand_aug,
#         gan_weight_path=weight_path,
#         use_gan=use_gan
#     )

#     # 步骤3：批量处理图像（无修改）
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
#     image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
#     logger.info(f"共找到 {len(image_files)} 张图像，开始全量增强...")

#     for img_path in image_files:
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             logger.warning(f"跳过无效图像 {img_path}：{e}")
#             continue

#         rel_dir = img_path.parent.relative_to(input_path)
#         target_dir = output_path / rel_dir
#         target_dir.mkdir(parents=True, exist_ok=True)

#         orig_path = target_dir / f"orig_{img_path.name}"
#         image.save(orig_path)

#         for i in range(augmentations_per_image):
#             use_gan_flag = use_gan and augmenter.gan_available and random.random() < 1
#             aug_imgs = augmenter.augment(image, use_gan=use_gan_flag)
            
#             for j, aug_img in enumerate(aug_imgs):
#                 if i * 2 + j >= augmentations_per_image:
#                     break
#                 aug_save_path = target_dir / f"aug_{i}_{j}_{img_path.name}"
#                 aug_img.save(aug_save_path)

#     logger.info(f"全量数据增强完成！结果保存至 {output_dir}")

# import os 
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from pathlib import Path
# import logging
# from tqdm import tqdm
# import subprocess
# import sys

# def install_package(package):
#     """在Python代码中安装指定包"""
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # 安装diffusers及其依赖
# try:
#     # 先尝试导入，判断是否已安装
#     import diffusers
#     import transformers
#     import accelerate
#     print("diffusers及其依赖已安装，无需重复安装")
# except ImportError:
#     print("正在安装diffusers及其依赖...")
#     install_package("diffusers")
#     install_package("transformers")
#     install_package("accelerate")
#     print("安装完成")

# # 后续可以正常使用diffusers相关功能

# # -------------------------- Stable Diffusion依赖 --------------------------
# from diffusers import StableDiffusionImg2ImgPipeline

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
# logger = logging.getLogger(__name__)

# # 设备设置
# def get_device():
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")

# device = get_device()
# logger.info(f"使用设备: {device}")


# # ===================== 1. 基础工具函数（Mixup/CutMix） =====================
# def mixup_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     if y is None:
#         return mixed_x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return mixed_x, y_a, y_b, lam

# def cutmix_data(x, y, alpha=0.2):
#     if alpha <= 0 or not isinstance(x, torch.Tensor):
#         return x, y, None, None
#     lam = np.random.beta(alpha, alpha)
#     batch_size, _, H, W = x.size()
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
#     index = torch.randperm(batch_size).to(x.device)
#     x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
#     if y is None:
#         return x, None, None, lam
#     else:
#         y_a, y_b = y, y[index]
#         return x, y_a, y_b, lam


# # ===================== 2. 数据加载器 =====================
# class ImageDataset(Dataset):
#     def __init__(
#         self, 
#         root_dir, 
#         img_size=(32, 32), 
#         use_auto_aug=False,
#         use_rand_aug=False,
#         is_train=True
#     ):
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.is_train = is_train
#         self.image_paths = [
#             p for p in Path(root_dir).glob('**/*') 
#             if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')
#         ]
#         logger.info(f"数据集加载完成，共 {len(self.image_paths)} 张图像（训练模式：{is_train}）")
#         self.transform = self._build_transform(use_auto_aug, use_rand_aug)

#     def _build_transform(self, use_auto_aug, use_rand_aug):
#         transform_list = []
#         if self.is_train:
#             transform_list.extend([
#                 transforms.RandomResizedCrop(size=self.img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.2),
#                 transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#                 transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#                 transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=0.2),
#                 transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
#             ])
#         transform_list.extend([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#         return transforms.Compose(transform_list)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             image = Image.open(img_path).convert('RGB')
#             return self.transform(image)
#         except Exception as e:
#             logger.warning(f"跳过损坏图像 {img_path}：{e}")
#             return torch.randn(3, self.img_size[0], self.img_size[1])


# # ===================== 3. GAN模型定义 =====================
# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         batch_size, C, w, h = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, w*h).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, w*h)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(batch_size, -1, w*h)
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, w, h)
#         return self.gamma * out + x


# class Generator(nn.Module):
#     def __init__(self, channels=3, img_size=32, input_dim=100):
#         super().__init__()
#         self.init_size = img_size // 4
#         self.input_dim = input_dim
#         self.l1 = nn.Sequential(nn.Linear(self.input_dim, 128 * self.init_size ** 2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             SelfAttention(64),
            
#             nn.Conv2d(64, channels, 3, 1, 1),
#             nn.Tanh()
#         )

#     def forward(self):
#         x = torch.randn(1, self.input_dim, device=device)
#         out = self.l1(x)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         return self.conv_blocks(out)


# class Discriminator(nn.Module):
#     def __init__(self, channels=3, img_size=32):
#         super().__init__()
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(channels, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         ds_size = img_size // (2**4)
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         return self.adv_layer(out)


# # ===================== 4. GAN训练器 =====================
# class GANTrainer:
#     def __init__(
#         self,
#         data_dir,
#         img_size=(32, 32),
#         epochs=30,
#         batch_size=64,
#         lr=0.0002,
#         weight_path="generator_weights.pth",
#         use_mixup=True,
#         use_cutmix=True,
#         generator_input_dim=100
#     ):
#         self.data_dir = data_dir
#         self.img_size = img_size
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_path = weight_path
#         self.use_mixup = use_mixup
#         self.use_cutmix = use_cutmix
#         self.generator_input_dim = generator_input_dim

#         self.generator = Generator(
#             img_size=img_size[0],
#             input_dim=self.generator_input_dim
#         ).to(device)
#         self.discriminator = Discriminator(img_size=img_size[0]).to(device)

#         self.adversarial_loss = nn.BCELoss().to(device)
#         self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
#         self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

#         self.dataset = ImageDataset(
#             root_dir=data_dir,
#             img_size=img_size,
#             use_rand_aug=True,
#             is_train=True
#         )
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True
#         )

#     def train(self):
#         logger.info(f"开始GAN训练（Mixup: {self.use_mixup}, CutMix: {self.use_cutmix}），共 {self.epochs} 轮")
        
#         for epoch in range(self.epochs):
#             pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
#             for imgs in pbar:
#                 batch_size = imgs.size(0)
#                 valid = torch.ones(batch_size, 1).to(device)
#                 fake = torch.zeros(batch_size, 1).to(device)
#                 real_imgs = imgs.to(device)

#                 if self.use_mixup:
#                     real_imgs, _, _, _ = mixup_data(real_imgs, None, alpha=0.1)
#                 if self.use_cutmix and not self.use_mixup:
#                     real_imgs, _, _, _ = cutmix_data(real_imgs, None, alpha=0.1)

#                 # 训练生成器
#                 self.optimizer_G.zero_grad()
#                 gen_imgs = torch.cat([self.generator() for _ in range(batch_size)], dim=0)
#                 g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
#                 g_loss.backward()
#                 self.optimizer_G.step()

#                 # 训练判别器
#                 self.optimizer_D.zero_grad()
#                 real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
#                 fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 d_loss = (real_loss + fake_loss) / 2
#                 d_loss.backward()
#                 self.optimizer_D.step()

#                 pbar.set_postfix({"D损失": d_loss.item(), "G损失": g_loss.item()})

#             if (epoch + 1) % 10 == 0:
#                 torch.save({
#                     "generator_state_dict": self.generator.state_dict(),
#                     "input_dim": self.generator_input_dim
#                 }, f"generator_weights_epoch_{epoch+1}.pth")
#                 logger.info(f"已保存第 {epoch+1} 轮GAN权重")

#         torch.save({
#             "generator_state_dict": self.generator.state_dict(),
#             "input_dim": self.generator_input_dim
#         }, self.weight_path)
#         logger.info(f"GAN训练完成，最终权重保存至 {self.weight_path}")


# # ===================== 5. 完整增强器（含Stable Diffusion） =====================
# class FullAugmenter:
#     def __init__(
#         self,
#         img_size=(32, 32),
#         use_auto_aug=False,
#         use_rand_aug=True,
#         gan_weight_path="generator_weights.pth",
#         use_gan=True,
#         # Stable Diffusion参数
#         use_sd=True,
#         sd_model_name="CompVis/stable-diffusion-v1-4",
#         sd_guidance_scale=7.5,
#         sd_strength=0.3,
#         sd_num_inference_steps=20
#     ):
#         self.img_size = img_size
#         self.use_gan = use_gan
        
#         # Stable Diffusion初始化
#         self.use_sd = use_sd
#         self.sd_guidance_scale = sd_guidance_scale
#         self.sd_strength = sd_strength
#         self.sd_num_inference_steps = sd_num_inference_steps
#         self.sd_pipeline = None
#         self.sd_available = self._init_stable_diffusion(sd_model_name)

#         # 传统增强
#         self.traditional_transform = self._build_traditional_transform(use_auto_aug, use_rand_aug)
#         self.inv_transform = transforms.Compose([
#             transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
#             transforms.ToPILImage()
#         ])

#         # GAN初始化
#         self.gan_available = False
#         self.generator = None
#         self.generator_input_dim = 100
#         if use_gan:
#             self.gan_available = self._load_gan_weights(gan_weight_path)
#             if self.gan_available:
#                 self.gan_preprocess = transforms.Compose([
#                     transforms.Resize(img_size),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#                 ])

#     def _init_stable_diffusion(self, model_name):
#         """初始化Stable Diffusion管道"""
#         if not self.use_sd:
#             logger.info("未启用Stable Diffusion增强")
#             return False
#         try:
#             # 加载SD模型（自动下载至缓存）
#             self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
#                 model_name,
#                 torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
#             ).to(device)
#             # 启用安全检查器（可选，避免生成不当内容）
#             self.sd_pipeline.safety_checker = lambda images, **kwargs: (images, [False]*len(images))
#             logger.info(f"成功加载Stable Diffusion模型：{model_name}")
#             return True
#         except Exception as e:
#             logger.error(f"Stable Diffusion初始化失败：{e}")
#             return False

#     def _load_gan_weights(self, weight_path):
#         """加载GAN权重"""
#         if not os.path.exists(weight_path):
#             logger.warning(f"未找到GAN权重文件：{weight_path}")
#             return False
#         try:
#             checkpoint = torch.load(weight_path, map_location=device)
#             self.generator_input_dim = checkpoint.get("input_dim", 100)
#             self.generator = Generator(
#                 img_size=self.img_size[0],
#                 input_dim=self.generator_input_dim
#             ).to(device)
#             self.generator.load_state_dict(checkpoint["generator_state_dict"])
#             self.generator.eval()
#             logger.info(f"成功加载GAN权重：{weight_path}")
#             return True
#         except Exception as e:
#             logger.error(f"GAN权重加载失败：{e}")
#             return False

#     def _build_traditional_transform(self, use_auto_aug, use_rand_aug):
#         """传统+自动化增强"""
#         transform_list = [
#             transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(degrees=(-15, 15), fill=(255, 255, 255)),
#             transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5, fill=(255, 255, 255)),
#             transforms.RandAugment(num_ops=2, magnitude=9) if use_rand_aug and not use_auto_aug else transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET) if use_auto_aug else transforms.Lambda(lambda x: x),
#             transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
#             transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ]
#         return transforms.Compose(transform_list)

#     def traditional_augment(self, image: Image.Image) -> Image.Image:
#         """传统增强"""
#         img_tensor = self.traditional_transform(image)
#         return self.inv_transform(img_tensor)

#     def gan_augment(self, image: Image.Image) -> Image.Image | None:
#         """GAN增强"""
#         if not self.gan_available or self.generator is None:
#             return None
#         with torch.no_grad():
#             gen_img = self.generator()
#             img_tensor = self.gan_preprocess(image).unsqueeze(0).to(device)
#             fused_img = 0.6 * gen_img + 0.4 * img_tensor
#             return self.inv_transform(fused_img.squeeze(0).cpu())

#     def sd_augment(self, image: Image.Image) -> Image.Image | None:
#         """Stable Diffusion增强（img2img）"""
#         if not self.sd_available or self.sd_pipeline is None:
#             return None
#         try:
#             # 调整图像尺寸以适配SD（SD对512x512效果最佳）
#             sd_img_size = (512, 512)
#             resized_img = image.resize(sd_img_size, Image.LANCZOS)
            
#             # 生成提示词（可根据数据集特性修改）
#             prompt = "high quality, clear details, consistent style"
            
#             # 调用img2img生成增强图像
#             with torch.no_grad():
#                 gen_images = self.sd_pipeline(
#                     prompt=prompt,
#                     image=resized_img,
#                     strength=self.sd_strength,
#                     guidance_scale=self.sd_guidance_scale,
#                     num_inference_steps=self.sd_num_inference_steps
#                 ).images
            
#             # 缩放回目标尺寸
#             return gen_images[0].resize(self.img_size, Image.LANCZOS)
#         except Exception as e:
#             logger.warning(f"Stable Diffusion增强失败：{e}")
#             return None

#     def augment(self, image: Image.Image, use_gan: bool = True, use_sd: bool = True) -> list[Image.Image]:
#         """综合增强接口"""
#         aug_imgs = [self.traditional_augment(image)]
        
#         # 添加GAN增强
#         if use_gan and self.gan_available and random.random() < 0.5:
#             gan_img = self.gan_augment(image)
#             if gan_img:
#                 aug_imgs.append(gan_img)
        
#         # 添加Stable Diffusion增强
#         if use_sd and self.sd_available and random.random() < 0.3:  # 较低概率，避免生成过多
#             sd_img = self.sd_augment(image)
#             if sd_img:
#                 aug_imgs.append(sd_img)
        
#         return aug_imgs


# # ===================== 6. 主流程：GAN训练+全量数据增强 =====================
# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 10,
#     img_size: tuple = (32, 32),
#     train_gan: bool = True,
#     gan_epochs: int = 30,
#     use_auto_aug: bool = True,
#     use_rand_aug: bool = True,
#     use_gan: bool = True,
#     use_sd: bool = True,
#     use_mixup_in_gan: bool = True,
#     generator_input_dim=100
# ):
#     weight_path = "generator_weights.pth"

#     # 步骤1：训练GAN
#     if train_gan or not os.path.exists(weight_path):
#         trainer = GANTrainer(
#             data_dir=input_dir,
#             img_size=img_size,
#             epochs=gan_epochs,
#             use_mixup=use_mixup_in_gan,
#             weight_path=weight_path,
#             generator_input_dim=generator_input_dim
#         )
#         trainer.train()
#     else:
#         logger.info("检测到已有GAN权重，跳过训练")

#     # 步骤2：初始化增强器
#     augmenter = FullAugmenter(
#         img_size=img_size,
#         use_auto_aug=use_auto_aug,
#         use_rand_aug=use_rand_aug,
#         gan_weight_path=weight_path,
#         use_gan=use_gan,
#         use_sd=use_sd
#     )

#     # 步骤3：批量处理图像
#     input_path = Path(input_dir)
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
#     image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
#     logger.info(f"共找到 {len(image_files)} 张图像，开始全量增强...")

#     for img_path in image_files:
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             logger.warning(f"跳过无效图像 {img_path}：{e}")
#             continue

#         rel_dir = img_path.parent.relative_to(input_path)
#         target_dir = output_path / rel_dir
#         target_dir.mkdir(parents=True, exist_ok=True)

#         # 保存原图
#         orig_path = target_dir / f"orig_{img_path.name}"
#         image.save(orig_path)

#         # 生成增强图像
#         for i in range(augmentations_per_image):
#             aug_imgs = augmenter.augment(image, use_gan=use_gan, use_sd=use_sd)
#             for j, aug_img in enumerate(aug_imgs):
#                 if i * len(aug_imgs) + j >= augmentations_per_image:
#                     break
#                 aug_save_path = target_dir / f"aug_{i}_{j}_{img_path.name}"
#                 aug_img.save(aug_save_path)

#     logger.info(f"全量数据增强完成！结果保存至 {output_dir}")
# import torch
# import torchvision.transforms as transforms
# from torchvision.transforms import autoaugment, transforms
# import numpy as np
# import random
# import os
# from PIL import Image
# import torchvision.transforms.functional as F

# class CIFAR100AdvancedAugmentation:
#     def __init__(self):
#         # 训练集增强 - 强增强策略
#         self.train_transform = transforms.Compose([
#             transforms.Resize(72),
#             transforms.RandomCrop(64, padding=4),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.2),
#             transforms.ColorJitter(
#                 brightness=0.4,
#                 contrast=0.4, 
#                 saturation=0.4,
#                 hue=0.1
#             ),
#             transforms.RandomRotation(15),
#             transforms.RandomGrayscale(p=0.1),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5071, 0.4867, 0.4408],
#                 std=[0.2675, 0.2565, 0.2761]
#             ),
#             transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
#         ])
        
#         # 验证集增强 - 仅基础预处理
#         self.val_transform = transforms.Compose([
#             transforms.Resize(64),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5071, 0.4867, 0.4408],
#                 std=[0.2675, 0.2565, 0.2761]
#             )
#         ])
        
#         # 测试时增强 (TTA)
#         self.tta_transform = transforms.Compose([
#             transforms.Resize(64),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5071, 0.4867, 0.4408],
#                 std=[0.2675, 0.2565, 0.2761]
#             )
#         ])

#     def get_train_transform(self):
#         return self.train_transform

#     def get_val_transform(self):
#         return self.val_transform

#     def get_tta_transforms(self):
#         """返回测试时增强的变换列表"""
#         tta_transforms = []
        
#         # 原始图像
#         tta_transforms.append(transforms.Compose([
#             transforms.Resize(64),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5071, 0.4867, 0.4408],
#                 std=[0.2675, 0.2565, 0.2761]
#             )
#         ]))
        
#         # 水平翻转
#         tta_transforms.append(transforms.Compose([
#             transforms.Resize(64),
#             transforms.RandomHorizontalFlip(p=1.0),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5071, 0.4867, 0.4408],
#                 std=[0.2675, 0.2565, 0.2761]
#             )
#         ]))
        
#         # 中心裁剪
#         tta_transforms.append(transforms.Compose([
#             transforms.Resize(72),
#             transforms.CenterCrop(64),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.5071, 0.4867, 0.4408],
#                 std=[0.2675, 0.2565, 0.2761]
#             )
#         ]))
        
#         return tta_transforms

# class CutMix:
#     def __init__(self, alpha=1.0):
#         self.alpha = alpha

#     def __call__(self, batch):
#         images, labels = batch
#         indices = torch.randperm(images.size(0))
#         shuffled_images = images[indices]
#         shuffled_labels = labels[indices]
        
#         lam = np.random.beta(self.alpha, self.alpha)
#         bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
#         images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
#         return images, labels, shuffled_labels, lam

#     def rand_bbox(self, size, lam):
#         W = size[2]
#         H = size[3]
#         cut_rat = np.sqrt(1. - lam)
#         cut_w = int(W * cut_rat)
#         cut_h = int(H * cut_rat)

#         cx = np.random.randint(W)
#         cy = np.random.randint(H)

#         bbx1 = np.clip(cx - cut_w // 2, 0, W)
#         bby1 = np.clip(cy - cut_h // 2, 0, H)
#         bbx2 = np.clip(cx + cut_w // 2, 0, W)
#         bby2 = np.clip(cy + cut_h // 2, 0, H)

#         return bbx1, bby1, bbx2, bby2

# class MixUp:
#     def __init__(self, alpha=1.0):
#         self.alpha = alpha

#     def __call__(self, batch):
#         images, labels = batch
#         indices = torch.randperm(images.size(0))
#         shuffled_images = images[indices]
#         shuffled_labels = labels[indices]
        
#         lam = np.random.beta(self.alpha, self.alpha)
#         images = lam * images + (1 - lam) * shuffled_images
        
#         return images, labels, shuffled_labels, lam

# def augment_dataset(input_dir, output_dir, augmentations_per_image=5):
#     """
#     对数据集进行增强，生成增强后的图像并保存到指定目录
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 定义增强变换
#     augmentation_transforms = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=15),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     ])
    
#     # 遍历每个类别目录
#     for class_name in os.listdir(input_dir):
#         class_input_dir = os.path.join(input_dir, class_name)
#         class_output_dir = os.path.join(output_dir, class_name)
#         os.makedirs(class_output_dir, exist_ok=True)
        
#         # 遍历类别目录中的每个图像
#         for img_name in os.listdir(class_input_dir):
#             img_path = os.path.join(class_input_dir, img_name)
            
#             # 打开图像
#             try:
#                 image = Image.open(img_path)
#             except Exception as e:
#                 print(f"Error opening image {img_path}: {e}")
#                 continue
            
#             # 保存原始图像
#             base_name = os.path.splitext(img_name)[0]
#             image.save(os.path.join(class_output_dir, f"{base_name}_0.jpg"))
            
#             # 生成增强图像
#             for i in range(augmentations_per_image):
#                 augmented_image = augmentation_transforms(image)
#                 augmented_image.save(os.path.join(class_output_dir, f"{base_name}_{i+1}.jpg"))
    
#     print(f"Data augmentation completed. Augmented images saved to {output_dir}")

# def load_transforms():
#     """
#     返回用于训练和验证的数据变换
#     """
#     augmentation = CIFAR100AdvancedAugmentation()
#     return {
#         'train': augmentation.get_train_transform(),
#         'val': augmentation.get_val_transform()
#     }
# import logging
# import random
# from pathlib import Path
# from typing import List, Tuple

# import numpy as np
# import torch
# import albumentations as A
# from PIL import Image

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class ImageAugmenter:
#     """Class to handle image augmentation operations using Albumentations."""

#     def __init__(
#         self,
#         augmentations_per_image: int = 5,
#         seed: int = 42,
#         save_original: bool = True,
#         image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
#     ):
#         """
#         Initialize the ImageAugmenter.

#         Args:
#             augmentations_per_image: Number of augmented versions per original image.
#             seed: Random seed for reproducibility.
#             save_original: Whether to save the original image with prefix 'orig_'.
#             image_extensions: Tuple of valid image file extensions.
#         """
#         self.augmentations_per_image = augmentations_per_image
#         self.seed = seed
#         self.save_original = save_original
#         self.image_extensions = image_extensions

#         self._set_seed()

#         # Define Albumentations pipeline
#         self.transform = A.Compose(
#             [
#                 A.Rotate(limit=15, p=0.8),
#                 A.HorizontalFlip(p=0.5),
#                 A.ShiftScaleRotate(
#                     shift_limit=0.1,
#                     scale_limit=0.1,
#                     rotate_limit=0,
#                     p=0.8,
#                     border_mode=0,  # cv2.BORDER_CONSTANT
#                 ),
#                 A.ColorJitter(
#                     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
#                 ),
#                 A.OneOf(
#                     [
#                         A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#                         A.MotionBlur(blur_limit=7, p=0.5),
#                     ],
#                     p=0.3,
#                 ),
#                 A.RandomBrightnessContrast(p=0.2),
#             ]
#         )

#     def _set_seed(self):
#         """Set random seeds for reproducibility."""
#         random.seed(self.seed)
#         np.random.seed(self.seed)
#         torch.manual_seed(self.seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(self.seed)

#     def augment_image(self, image: Image.Image) -> Image.Image:
#         """
#         Apply augmentation transforms to a single image using Albumentations.

#         Args:
#             image: PIL Image to augment.

#         Returns:
#             Augmented PIL Image.
#         """
#         # Convert PIL to NumPy array (RGB)
#         image_np = np.array(image)

#         # Apply Albumentations transform
#         augmented = self.transform(image=image_np)
#         augmented_image_np = augmented["image"]

#         # Convert back to PIL Image
#         return Image.fromarray(augmented_image_np.astype(np.uint8))

#     def process_directory(self, input_dir: str, output_dir: str) -> None:
#         """
#         Augment all images in input directory and save to output directory.

#         Preserves folder structure. Skips files that fail to load.

#         Args:
#             input_dir: Path to input directory with class subfolders.
#             output_dir: Path to output directory for augmented images.
#         """
#         input_path = Path(input_dir)
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
#         count = 0

#         image_files = self._find_image_files(input_path)

#         logger.info(f"Found {len(image_files)} images to augment.")

#         for img_path in image_files:
#             try:
#                 image = Image.open(img_path).convert("RGB")
#             except Exception as e:
#                 logger.warning(f"Failed to load image {img_path}: {e}")
#                 continue

#             # Determine output subdirectory
#             rel_dir = img_path.parent.relative_to(input_path)
#             target_dir = output_path / rel_dir
#             if not target_dir.exists():
#                 target_dir.mkdir(parents=True, exist_ok=True)

#             # Save original if requested
#             if self.save_original:
#                 orig_name = f"orig_{img_path.name}"
#                 image.save(target_dir / orig_name)

#             # Generate and save augmented versions
#             for i in range(self.augmentations_per_image):
#                 augmented = self.augment_image(image.copy())
#                 aug_name = f"aug_{i}_{img_path.name}"
#                 augmented.save(target_dir / aug_name)
#                 count += 1

#         logger.info(
#             f"Augmentation of {count} images completed. Output saved to: {output_dir}"
#         )

#     def _find_image_files(self, root: Path) -> List[Path]:
#         """
#         Recursively find all image files in directory.

#         Args:
#             root: Root directory path.

#         Returns:
#             List of image file paths.
#         """
#         files = []
#         for ext in self.image_extensions:
#             files.extend(root.rglob(f"*{ext}"))
#         return files


# def augment_dataset(
#     input_dir: str,
#     output_dir: str,
#     augmentations_per_image: int = 5,
#     seed: int = 42,
# ) -> None:
#     """
#     Backward-compatible wrapper for legacy code.

#     Args:
#         input_dir: Directory containing cleaned images (organized by class).
#         output_dir: Directory to save augmented images.
#         augmentations_per_image: Number of augmented versions per original image.
#         seed: Random seed for reproducibility.
#     """
#     augmenter = ImageAugmenter(
#         augmentations_per_image=augmentations_per_image, seed=seed, save_original=True
#     )
#     augmenter.process_directory(input_dir, output_dir)




# import torch
# import torchvision.transforms as transforms
# from torchvision.transforms import autoaugment, transforms
# import numpy as np
# import random
# import os
# from PIL import Image
# import torchvision.transforms.functional as F

# class ProgressiveLearning:
#     """渐进式学习：从简单到复杂的训练策略"""
#     def __init__(self, image_sizes=[32, 48, 64, 80], epochs_per_stage=[20, 30, 40, 10]):
#         self.image_sizes = image_sizes
#         self.epochs_per_stage = epochs_per_stage
#         self.current_stage = 0
        
#     def get_current_transform(self, is_train=True):
#         size = self.image_sizes[self.current_stage]
#         if is_train:
#             return transforms.Compose([
#                 transforms.Resize((size + 8, size + 8)),
#                 transforms.RandomCrop(size, padding=4),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(15),
#                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#                 transforms.RandomGrayscale(p=0.1),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
#                 transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
#             ])
#         else:
#             return transforms.Compose([
#                 transforms.Resize((size, size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
#             ])
    
#     def should_advance_stage(self, epoch, total_epochs):
#         cumulative_epochs = sum(self.epochs_per_stage[:self.current_stage+1])
#         return epoch >= cumulative_epochs and self.current_stage < len(self.image_sizes)-1
    
#     def advance_stage(self):
#         self.current_stage += 1
#         print(f"渐进式学习: 进入阶段 {self.current_stage}, 图像尺寸: {self.image_sizes[self.current_stage]}")

# class CIFAR100AdvancedAugmentation:
#     def __init__(self, progressive_learning=None):
#         self.progressive_learning = progressive_learning
        
#         # 标准增强（如果不用渐进式学习）
#         self.standard_train_transform = transforms.Compose([
#             transforms.Resize(72),
#             transforms.RandomCrop(64, padding=4),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(15),
#             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#             transforms.RandomGrayscale(p=0.1),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
#             transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
#         ])
        
#         self.standard_val_transform = transforms.Compose([
#             transforms.Resize(64),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
#         ])

#     def get_train_transform(self):
#         if self.progressive_learning:
#             return self.progressive_learning.get_current_transform(is_train=True)
#         return self.standard_train_transform

#     def get_val_transform(self):
#         if self.progressive_learning:
#             return self.progressive_learning.get_current_transform(is_train=False)
#         return self.standard_val_transform

# def augment_dataset(input_dir, output_dir, augmentations_per_image=5):
#     """
#     对数据集进行增强，生成增强后的图像并保存到指定目录
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 定义增强变换
#     augmentation_transforms = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=15),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     ])
    
#     # 遍历每个类别目录
#     for class_name in os.listdir(input_dir):
#         class_input_dir = os.path.join(input_dir, class_name)
#         class_output_dir = os.path.join(output_dir, class_name)
#         os.makedirs(class_output_dir, exist_ok=True)
        
#         # 遍历类别目录中的每个图像
#         for img_name in os.listdir(class_input_dir):
#             img_path = os.path.join(class_input_dir, img_name)
            
#             # 打开图像
#             try:
#                 image = Image.open(img_path)
#             except Exception as e:
#                 print(f"Error opening image {img_path}: {e}")
#                 continue
            
#             # 保存原始图像
#             base_name = os.path.splitext(img_name)[0]
#             image.save(os.path.join(class_output_dir, f"{base_name}_0.jpg"))
            
#             # 生成增强图像
#             for i in range(augmentations_per_image):
#                 augmented_image = augmentation_transforms(image)
#                 augmented_image.save(os.path.join(class_output_dir, f"{base_name}_{i+1}.jpg"))
    
#     print(f"Data augmentation completed. Augmented images saved to {output_dir}")

# def load_transforms(progressive_learning=None):
#     """
#     返回用于训练和验证的数据变换
#     """
#     augmentation = CIFAR100AdvancedAugmentation(progressive_learning)
#     return {
#         'train': augmentation.get_train_transform(),
#         'val': augmentation.get_val_transform()
#     }

import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import albumentations as A
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    """Class to handle image augmentation operations using Albumentations."""

    def __init__(
        self,
        augmentations_per_image: int = 5,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        """
        Initialize the ImageAugmenter.

        Args:
            augmentations_per_image: Number of augmented versions per original image.
            seed: Random seed for reproducibility.
            save_original: Whether to save the original image with prefix 'orig_'.
            image_extensions: Tuple of valid image file extensions.
        """
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions

        self._set_seed()

        # Define Albumentations pipeline
        self.transform = A.Compose(
            [
                A.Rotate(limit=15, p=0.8),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=0,
                    p=0.8,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MotionBlur(blur_limit=7, p=0.5),
                    ],
                    p=0.3,
                ),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation transforms to a single image using Albumentations.

        Args:
            image: PIL Image to augment.

        Returns:
            Augmented PIL Image.
        """
        # Convert PIL to NumPy array (RGB)
        image_np = np.array(image)

        # Apply Albumentations transform
        augmented = self.transform(image=image_np)
        augmented_image_np = augmented["image"]

        # Convert back to PIL Image
        return Image.fromarray(augmented_image_np.astype(np.uint8))

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Augment all images in input directory and save to output directory.

        Preserves folder structure. Skips files that fail to load.

        Args:
            input_dir: Path to input directory with class subfolders.
            output_dir: Path to output directory for augmented images.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0

        image_files = self._find_image_files(input_path)

        logger.info(f"Found {len(image_files)} images to augment.")

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            # Determine output subdirectory
            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            # Save original if requested
            if self.save_original:
                orig_name = f"orig_{img_path.name}"
                image.save(target_dir / orig_name)

            # Generate and save augmented versions
            for i in range(self.augmentations_per_image):
                augmented = self.augment_image(image.copy())
                aug_name = f"aug_{i}_{img_path.name}"
                augmented.save(target_dir / aug_name)
                count += 1

        logger.info(
            f"Augmentation of {count} images completed. Output saved to: {output_dir}"
        )

    def _find_image_files(self, root: Path) -> List[Path]:
        """
        Recursively find all image files in directory.

        Args:
            root: Root directory path.

        Returns:
            List of image file paths.
        """
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5,
    seed: int = 42,
) -> None:
    """
    Backward-compatible wrapper for legacy code.

    Args:
        input_dir: Directory containing cleaned images (organized by class).
        output_dir: Directory to save augmented images.
        augmentations_per_image: Number of augmented versions per original image.
        seed: Random seed for reproducibility.
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image, seed=seed, save_original=True
    )
    augmenter.process_directory(input_dir, output_dir)