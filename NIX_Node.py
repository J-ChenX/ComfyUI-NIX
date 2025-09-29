# ==== 标准库 ====
import json
import math
import os
import re
import sys

# 在导入 comfy.* 之前，确保将本插件内的 comfy 子目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

# ==== 第三方库 ====
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo

# ==== 项目/本地模块（ComfyUI 相关）====
import comfy.utils
import comfy.model_management
from comfy.cli_args import args
import folder_paths

# 根据环境兼容导入 IO，并提供 ANY_T 兜底
try:
    from comfy.comfy_types.node_typing import IO
    ANY_T = IO.ANY
except Exception:
    IO = None
    ANY_T = "*"

# ==== 运行期辅助函数 ====
def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()


def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)




class NIX_PathLoading:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
                "mask_path": ("STRING", {"default": ""}),
                "channel": (["red", "blue", "green", "purple", "yellow", "cyan", "white"], {"default": "white"}),
            },
            "optional": {
                # load_quantity: 0 表示加载全部；默认 1
                "load_quantity": ("INT", {"default": 1, "min": 0, "step": 1}),
                # start_index: 从 1 开始计数；默认 1
                "start_index": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "quantity_number")
    OUTPUT_IS_LIST = (True, True, True, False)

    FUNCTION = "pathloading"
    CATEGORY = "NIX"

    @classmethod
    def pathloading(cls, image_path: str, mask_path: str, channel: str,
                    load_quantity: int = 1, start_index: int = 1):
        if not image_path and not mask_path:
            raise FileNotFoundError("图片路径和蒙版路径均为空")

        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']

        # 加载并过滤文件列表（自然排序 + 1 基下标 start_index）
        image_files = cls.load_files(image_path, valid_extensions, start_index)
        mask_files = cls.load_files(mask_path, valid_extensions, start_index)

        if not image_files and not mask_files:
            raise FileNotFoundError("没有找到任何有效的图片或蒙版文件")

        # 仅有图片
        if image_files and not mask_files:
            # 计算最大可加载数量（不受 load_quantity 影响）
            max_count = len(image_files)
            quantity_number = ";".join(str(i) for i in range(1, max_count + 1))

            images, filenames = cls.load_images(image_files, load_quantity, remove_file_extensions=False)
            if not images:
                raise FileNotFoundError("没有加载到任何图片")
            h, w = images[0].shape[1], images[0].shape[2]
            masks = [torch.zeros((1, h, w), dtype=images[0].dtype, device=images[0].device) for _ in range(len(images))]
            return images, masks, filenames, quantity_number

        # 仅有蒙版
        if mask_files and not image_files:
            # 计算最大可加载数量（不受 load_quantity 影响）
            max_count = len(mask_files)
            quantity_number = ";".join(str(i) for i in range(1, max_count + 1))

            masks, filenames = cls.load_images(mask_files, load_quantity, remove_file_extensions=False,
                                               is_mask=True, channel=channel)
            if not masks:
                raise FileNotFoundError("没有加载到任何蒙版")
            h, w = masks[0].shape[1], masks[0].shape[2]
            images = [torch.zeros((1, h, w, 3), dtype=masks[0].dtype, device=masks[0].device) for _ in range(len(masks))]
            return images, masks, filenames, quantity_number

        # 同时存在图片与蒙版（按同 stem 匹配）
        common_files_sorted = sorted(
            set(image_files.keys()).intersection(mask_files.keys()),
            key=cls.natural_sort_key
        )
        # 计算最大可加载数量（不受 load_quantity 影响）
        max_count = len(common_files_sorted)
        quantity_number = ";".join(str(i) for i in range(1, max_count + 1))

        images, masks, filenames = cls.load_images_and_masks(image_files, mask_files, load_quantity, channel)
        if len(images) != len(masks):
            raise ValueError("图片与蒙版数量不匹配")
        return images, masks, filenames, quantity_number

    @staticmethod
    def natural_sort_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    @classmethod
    def load_files(cls, path, valid_extensions, start_index):
        if path and os.path.isdir(path):
            path = os.path.normpath(path)
            files = [f for f in os.listdir(path) if any(f.lower().endswith(ext) for ext in valid_extensions)]
            sorted_files = sorted(files, key=cls.natural_sort_key)

            mapping = {}
            # 使用 1 基下标进行遍历与过滤
            for idx, f in enumerate(sorted_files, start=1):
                if idx < start_index:
                    continue
                stem = os.path.splitext(f)[0]
                mapping[stem] = os.path.join(path, f)
            return mapping
        return {}

    @staticmethod
    def load_images(files, load_quantity, remove_file_extensions=False, is_mask=False, channel="red"):
        images = []
        filenames = []
        count = 0
        for filename, filepath in files.items():
            if load_quantity > 0 and count >= load_quantity:
                break
            img = NIX_PathLoading.process_image(filepath, is_mask=is_mask, channel=channel)
            images.append(img)
            filenames.append(filename)  # stem
            count += 1
        return images, filenames

    @classmethod
    def load_images_and_masks(cls, image_files, mask_files, load_quantity, channel):
        images, masks, filenames = [], [], []
        count = 0
        common_files = sorted(set(image_files.keys()).intersection(mask_files.keys()), key=cls.natural_sort_key)
        for filename in common_files:
            if load_quantity > 0 and count >= load_quantity:
                break
            img = NIX_PathLoading.process_image(image_files[filename], is_mask=False)
            msk = NIX_PathLoading.process_image(mask_files[filename], is_mask=True, channel=channel)
            images.append(img)
            masks.append(msk)
            filenames.append(filename)  # stem
            count += 1
        return images, masks, filenames

    @staticmethod
    def process_image(filepath, is_mask=False, channel="red"):
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img).convert("RGB")
        np_img = np.array(img).astype(np.float32) / 255.0

        if is_mask:
            t = torch.from_numpy(np_img)[None, :, :, :3]  # (1,H,W,3)
            mask = NIX_PathLoading.extract_channel(t, channel)  # (1,H,W)
            return mask
        else:
            return torch.from_numpy(np_img)[None, ...]  # (1,H,W,3)

    @staticmethod
    def extract_channel(img: torch.Tensor, channel: str) -> torch.Tensor:
        r = img[:, :, :, 0]
        g = img[:, :, :, 1]
        b = img[:, :, :, 2]
        if channel == "red":
            m = r - g - b
        elif channel == "green":
            m = g - r - b
        elif channel == "blue":
            m = b - r - g
        elif channel == "yellow":
            m = r * g - b
        elif channel == "purple":
            m = r * b - g
        elif channel == "cyan":
            m = g * b - r
        elif channel == "white":
            m = r * g * b
        else:
            m = r
        return m.clamp(0.0, 1.0)


class NIX_RotateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "angle": ("INT", {"default": 0, "min": -180, "max": 180, "step": 1}),
            },
            "optional": {
                "masks": ("MASK",),
                "flip_horizontal": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled", "default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "image_rotate"
    CATEGORY = "NIX"

    def image_rotate(self, images, angle, masks=None, flip_horizontal=False):
        rotated_images = []
        rotated_masks = []

        sampler_rgb = Image.BICUBIC
        sampler_mask = Image.BILINEAR

        batch = images.shape[0]
        for i in range(batch):
            image = self.tensor2pil(images[i])
            if flip_horizontal:
                image = ImageOps.mirror(image)
            w, h = image.size

            if angle != 0:
                max_dim = int(np.ceil(np.hypot(w, h)))
                canvas = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
                canvas.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
                rotated_image = canvas.rotate(angle, sampler_rgb, expand=True)
                cx = (rotated_image.width - max_dim) // 2
                cy = (rotated_image.height - max_dim) // 2
                rotated_image = rotated_image.crop((cx, cy, cx + max_dim, cy + max_dim))
                out_w, out_h = max_dim, max_dim
            else:
                rotated_image = image
                out_w, out_h = w, h

            rotated_images.append(self.pil2tensor(rotated_image))

            if masks is not None and i < masks.shape[0]:
                mask = self.tensor2pil(masks[i])
                if flip_horizontal:
                    mask = ImageOps.mirror(mask)
                if angle != 0:
                    mcanvas = Image.new("L", (max_dim, max_dim), 0)
                    mcanvas.paste(mask, ((max_dim - w) // 2, (max_dim - h) // 2))
                    rmask = mcanvas.rotate(angle, sampler_mask, expand=True)
                    rmask = rmask.crop((cx, cy, cx + max_dim, cy + max_dim))
                else:
                    rmask = mask
                rotated_masks.append(self.pil2tensor(rmask))
            else:
                rotated_masks.append(torch.zeros((1, out_h, out_w), dtype=images.dtype, device=images.device))

        rotated_images = torch.cat(rotated_images, dim=0)
        rotated_masks = torch.cat(rotated_masks, dim=0)
        return rotated_images, rotated_masks

    @staticmethod
    def tensor2pil(image):
        arr = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class NIX_MaskCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "expand": ("FLOAT", {"default": 0.2, "min": 0, "max": 101, "step": 0.01}),
                "blur": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "filling_method": (["filling", "move"], {"default": "filling"}),
            },
            "optional": {
                "original_image": ("IMAGE",),
                "mask": ("MASK",),
                "binarize": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled", "default": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "PIPE_LINE")
    RETURN_NAMES = ("crop_image", "crop_mask", "pipe")
    FUNCTION = "maskcrop"
    CATEGORY = "NIX"

    def maskcrop(self, width, height, expand, blur, filling_method, original_image=None, mask=None, binarize=False):
        if mask is None:
            raise ValueError("mask 不能为空")

        # 统一形状：mask (B,H,W)，image (B,H,W,3)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if original_image is None:
            # 用 mask 生成灰阶三通道图作为占位
            original_image = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # 对齐空间尺寸
        original_image, mask = self.match_size(original_image, mask)

        # 统一数值与类型
        mask = mask.float().clamp(0, 1)

        # 可选二值化
        if binarize:
            mask = self.binarize_mask(mask)

        # 对齐 batch 大小
        original_image, mask = self.align_batch_size(original_image, mask)

        # 计算 bbox（为空则使用整图）
        if torch.any(mask > 0):
            x1, x2, y1, y2 = self.get_bbox_coordinates(mask)
        else:
            x1, y1 = 0, 0
            x2, y2 = original_image.shape[2], original_image.shape[1]

        bbox_width = int(x2 - x1)
        bbox_height = int(y2 - y1)
        aspect_ratio = float(width) / float(height) if height > 0 else 1.0

        # 模糊（动态半径按 bbox 尺寸估算）
        if blur > 0:
            blur_radius = float(blur) * max(bbox_width, bbox_height) / 1024.0
            mask = self.gaussian_blur(mask, blur_radius)

        # 调整 bbox 尺寸以匹配目标比例并扩展
        new_width_adj, new_height_adj = self.adjust_bbox_size(bbox_width, bbox_height, aspect_ratio, expand, blur)

        # 将 bbox 居中到原中心
        x1, x2, y1, y2 = self.recenter_bbox(int(x1), int(x2), int(y1), int(y2), int(new_width_adj), int(new_height_adj))

        original_x1, original_y1 = int(x1), int(y1)

        # 裁剪并填充
        padded_image, padded_mask = self.crop_and_pad(original_image, mask, int(x1), int(x2), int(y1), int(y2),
                                                      int(new_width_adj), int(new_height_adj), filling_method)

        # 记录填充后的中间尺寸
        final_height, final_width = padded_image.shape[1], padded_image.shape[2]

        # 缩放到目标输出尺寸
        padded_image = self.upscale_image(padded_image, width, height)
        padded_mask = self.resize(padded_mask, width, height)

        # 计算贴回的最终坐标
        if filling_method == "move":
            final_x1, final_y1 = self.calculate_final_coordinates(
                original_x1, original_y1, original_image, int(new_width_adj), int(new_height_adj), filling_method
            )
        else:
            final_x1, final_y1 = original_x1, original_y1

        pipe = {
            "original_image": original_image,
            "crop_mask": padded_mask,
            "crop_width": int(final_width),
            "crop_height": int(final_height),
            "crop_X": int(final_x1),
            "crop_Y": int(final_y1),
        }

        return padded_image, padded_mask, pipe

    def calculate_final_coordinates(self, x1, y1, original_image, new_width_adj, new_height_adj, filling_method):
        if filling_method == "move":
            H, W = original_image.shape[1], original_image.shape[2]
            if new_width_adj > W or new_height_adj > H:
                crop_x1 = max(0, x1)
                crop_y1 = max(0, y1)
                mask_cropped_width = min(W, x1 + new_width_adj) - crop_x1
                mask_cropped_height = min(H, y1 + new_height_adj) - crop_y1
                x_offset = (new_width_adj - mask_cropped_width) // 2
                y_offset = (new_height_adj - mask_cropped_height) // 2
                return int(crop_x1 - x_offset), int(crop_y1 - y_offset)
            else:
                target_width = min(new_width_adj, W)
                target_height = min(new_height_adj, H)
                x1 = max(0, min(x1, W - target_width))
                y1 = max(0, min(y1, H - target_height))
                return int(x1), int(y1)
        return int(x1), int(y1)

    def match_size(self, image, mask):
        H, W = image.shape[1], image.shape[2]
        if mask.shape[1] != H or mask.shape[2] != W:
            m = mask.unsqueeze(1).float()
            m = torch.nn.functional.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
            mask = m.squeeze(1)
        return image, mask

    def binarize_mask(self, mask):
        return (mask > 0.5).float()

    def align_batch_size(self, image, mask):
        bi, bm = image.shape[0], mask.shape[0]
        if bi < bm:
            repeat_count = bm - bi
            image = torch.cat((image, image[-1:].repeat(repeat_count, 1, 1, 1)), dim=0)
        elif bi > bm:
            mask = torch.cat((mask, mask[-1:].repeat(bi - bm, 1, 1)), dim=0)
        return image, mask

    def get_bbox_coordinates(self, mask):
        idx = torch.nonzero(mask > 0, as_tuple=False)
        if idx.numel() == 0:
            return 0, mask.shape[2], 0, mask.shape[1]
        _, y, x = idx.t()
        x1, x2 = x.min().item(), x.max().item() + 1
        y1, y2 = y.min().item(), y.max().item() + 1
        return int(x1), int(x2), int(y1), int(y2)

    def gaussian_blur(self, mask, blur_radius: float):
        out = []
        for i in range(mask.shape[0]):
            m = mask[i].clamp(0, 1)
            m_np = (m.cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(m_np, mode="L")
            blurred = pil.filter(ImageFilter.GaussianBlur(float(blur_radius)))
            bl_t = torch.from_numpy(np.array(blurred)).float() / 255.0
            out.append(bl_t)
        out = torch.stack(out, dim=0).to(mask.device)
        return out.clamp(0, 1)

    def adjust_bbox_size(self, bbox_width, bbox_height, aspect_ratio, expand, blur):
        if bbox_height == 0:
            bbox_height = 1
        if bbox_width / bbox_height > aspect_ratio:
            new_height_adj = bbox_width / aspect_ratio
            new_width_adj = bbox_width
        else:
            new_width_adj = bbox_height * aspect_ratio
            new_height_adj = bbox_height

        new_width_adj = int(round(new_width_adj * (1.0 + float(expand))))
        new_height_adj = int(round(new_height_adj * (1.0 + float(expand))))

        blur_factor = float(blur) * max(bbox_width, bbox_height) / 1024.0
        if bbox_width / bbox_height > aspect_ratio and blur_factor > expand * new_width_adj / 2.0:
            new_width_adj = int(round(new_width_adj + 2 * blur - expand * new_width_adj + 2))
            new_height_adj = int(round(new_width_adj / aspect_ratio)) if aspect_ratio > 0 else new_height_adj
        elif blur_factor > expand * new_height_adj / 2.0:
            new_width_adj = int(round(new_width_adj + 2 * blur - expand * new_width_adj + 2))
            new_height_adj = int(round(new_width_adj / aspect_ratio)) if aspect_ratio > 0 else new_height_adj

        return max(1, new_width_adj), max(1, new_height_adj)

    def recenter_bbox(self, x1, x2, y1, y2, new_width_adj, new_height_adj):
        x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
        x1_new = x_center - new_width_adj // 2
        x2_new = x_center + new_width_adj // 2
        y1_new = y_center - new_height_adj // 2
        y2_new = y_center + new_height_adj // 2
        return int(x1_new), int(x2_new), int(y1_new), int(y2_new)

    def crop_and_pad(self, image, mask, x1, x2, y1, y2, new_width_adj, new_height_adj, filling_method):
        B, H, W = image.shape[0], image.shape[1], image.shape[2]
        gray_value = 0.5

        if filling_method == "filling":
            padded_mask = torch.zeros((B, new_height_adj, new_width_adj), dtype=mask.dtype, device=mask.device)
            padded_image = torch.ones((B, new_height_adj, new_width_adj, 3), dtype=image.dtype, device=image.device) * gray_value

            out_cx = new_width_adj // 2
            out_cy = new_height_adj // 2
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2

            offset_x = out_cx - bbox_cx
            offset_y = out_cy - bbox_cy

            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(W, x2)
            src_y2 = min(H, y2)

            dst_x1 = src_x1 + offset_x
            dst_y1 = src_y1 + offset_y
            dst_x2 = src_x2 + offset_x
            dst_y2 = src_y2 + offset_y

            dst_x1c = max(0, dst_x1)
            dst_y1c = max(0, dst_y1)
            dst_x2c = min(new_width_adj, dst_x2)
            dst_y2c = min(new_height_adj, dst_y2)

            if (dst_x1c < dst_x2c and dst_y1c < dst_y2c and src_x1 < src_x2 and src_y1 < src_y2):
                src_x1_adj = src_x1 + (dst_x1c - dst_x1)
                src_y1_adj = src_y1 + (dst_y1c - dst_y1)
                src_x2_adj = src_x1_adj + (dst_x2c - dst_x1c)
                src_y2_adj = src_y1_adj + (dst_y2c - dst_y1c)

                padded_mask[:, dst_y1c:dst_y2c, dst_x1c:dst_x2c] = mask[:, src_y1_adj:src_y2_adj, src_x1_adj:src_x2_adj]
                padded_image[:, dst_y1c:dst_y2c, dst_x1c:dst_x2c, :] = image[:, src_y1_adj:src_y2_adj, src_x1_adj:src_x2_adj, :]

        else:  # move
            if new_width_adj > W or new_height_adj > H:
                crop_x1 = max(0, x1)
                crop_x2 = min(W, x2)
                crop_y1 = max(0, y1)
                crop_y2 = min(H, y2)

                mask_cropped = mask[:, crop_y1:crop_y2, crop_x1:crop_x2]
                image_cropped = image[:, crop_y1:crop_y2, crop_x1:crop_x2, :]

                padded_mask = torch.zeros((B, new_height_adj, new_width_adj), dtype=mask.dtype, device=mask.device)
                padded_image = torch.ones((B, new_height_adj, new_width_adj, 3), dtype=image.dtype, device=image.device) * gray_value

                y_offset = (new_height_adj - mask_cropped.shape[1]) // 2
                x_offset = (new_width_adj - mask_cropped.shape[2]) // 2

                padded_mask[:, y_offset:y_offset + mask_cropped.shape[1], x_offset:x_offset + mask_cropped.shape[2]] = mask_cropped
                padded_image[:, y_offset:y_offset + image_cropped.shape[1], x_offset:x_offset + image_cropped.shape[2], :] = image_cropped
            else:
                target_width = min(new_width_adj, W)
                target_height = min(new_height_adj, H)

                if x1 < 0:
                    x1 = 0
                    x2 = target_width
                elif x2 > W:
                    x2 = W
                    x1 = max(0, x2 - target_width)

                if y1 < 0:
                    y1 = 0
                    y2 = target_height
                elif y2 > H:
                    y2 = H
                    y1 = max(0, y2 - target_height)

                if x2 - x1 < target_width and x2 < W:
                    x2 = min(W, x1 + target_width)
                if y2 - y1 < target_height and y2 < H:
                    y2 = min(H, y1 + target_height)

                padded_image = image[:, y1:y2, x1:x2, :]
                padded_mask = mask[:, y1:y2, x1:x2]

        return padded_image, padded_mask

    def upscale_image(self, image, width, height):
        image = image.movedim(-1, 1)  # (B,3,H,W)
        image = comfy.utils.common_upscale(image, width, height, "lanczos", "disabled")
        image = image.movedim(1, -1)  # (B,H,W,3)
        return image

    def resize(self, mask, width, height):
        m = mask.unsqueeze(1).float()
        m = torch.nn.functional.interpolate(m, size=(height, width), mode='bicubic', align_corners=False)
        return m.squeeze(1)


class NIX_ImageComposite:
    # 定义类方法，返回该类所需的输入类型和参数
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_image": ("IMAGE",),  # 源图像
                "pipe": ("PIPE_LINE",)  # 传入的 pipe 对象
            },
            "optional": {
                "original_image": ("IMAGE",)  # 可选输入的原始图像
            }
        }

    # 定义返回类型
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "NIX"

    def composite(self, sampled_image, pipe, original_image=None):
        # 如果 original_image 没有提供，则从 pipe 获取
        if original_image is None:
            original_image = pipe["original_image"]
        padded_mask = pipe["crop_mask"]
        final_width = pipe["crop_width"]
        final_height = pipe["crop_height"]
        x1 = pipe["crop_X"]
        y1 = pipe["crop_Y"]

        # 预处理图像，准备目标图像
        original_image = self.prepare_image(original_image)
        # 预处理图像，调整源图像大小
        sampled_image = self.prepare_image(self.upscale_image(sampled_image, final_width, final_height))

        # 在预处理蒙版前保护padded_mask
        sampled_mask = padded_mask
        sampled_mask = self.prepare_mask(sampled_image, sampled_mask)

        # 处理负值坐标，确保坐标在有效范围内
        x1, y1, sampled_image, sampled_mask = self.handle_negative_coordinates(x1, y1, sampled_image, sampled_mask)

        # 确保源图像和蒙版的数量与目标图像一致
        sampled_image, sampled_mask = self.align_batch_size(sampled_image, sampled_mask)

        # 计算可见宽度和高度，确定合成区域的大小
        visible_width, visible_height = self.calculate_visible_dimensions(original_image, sampled_image, x1, y1)

        # 调整蒙版大小并生成反蒙版
        sampled_mask, inverse_mask = self.adjust_mask(sampled_mask, visible_width, visible_height)

        # 分别计算源图像和目标图像的可见部分并进行合成
        composite_portion = self.compute_composite_portion(original_image, sampled_image, sampled_mask, inverse_mask,
                                                           visible_width, visible_height, x1, y1)

        # 将合成后的部分赋值给目标图像的指定区域
        original_image[:, :, y1:y1 + visible_height, x1:x1 + visible_width] = composite_portion

        # 恢复维度并返回结果
        return (original_image.movedim(1, -1),)

    def prepare_image(self, image):
        # 克隆图像并移动维度，使通道维度在第2维
        return image.clone().movedim(-1, 1)

    def upscale_image(self, image, width, height):
        # 调整图像维度，将其移动到所需维度
        image = image.movedim(-1, 1)
        # 使用指定算法（lanczos）进行上采样并禁用特定选项
        image = comfy.utils.common_upscale(image, width, height, "lanczos", "disabled")
        # 恢复图像维度
        return image.movedim(1, -1)

    def prepare_mask(self, image, mask):
        if mask is None:
            return torch.ones((image.shape[0], 1, image.shape[2], image.shape[3]),
                              dtype=image.dtype, device=image.device)
        mask = mask.to(image.device).float().clamp(0, 1)
        return torch.nn.functional.interpolate(
            mask.view(-1, 1, *mask.shape[-2:]),
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

    def handle_negative_coordinates(self, x, y, image, mask):
        # 处理负值x坐标，如果x为负，则调整源图像和蒙版
        if x < 0:
            image = image[:, :, :, -x:]
            mask = mask[:, :, :, -x:]
            x = 0
        # 处理负值y坐标，如果y为负，则调整源图像和蒙版
        if y < 0:
            image = image[:, :, -y:, :]
            mask = mask[:, :, -y:, :]
            y = 0
        return x, y, image, mask

    # 对齐蒙版和图像的数量
    def align_batch_size(self, image, mask):
        if image.shape[0] < mask.shape[0]:  # 如果图像数量小于蒙版数量
            image = self.repeat_last_image(image, mask.shape[0] - image.shape[0])  # 重复最后一张图像
        elif image.shape[0] > mask.shape[0]:  # 如果图像数量大于蒙版数量
            image = image[:mask.shape[0]]  # 重复最后一张蒙版
        return image, mask  # 返回对齐后的蒙版和图像

    def calculate_visible_dimensions(self, original_image, sampled_image, x, y):
        # 计算可见宽度和高度，确保不超出目标图像的边界
        visible_width = min(original_image.shape[3] - x, sampled_image.shape[3])
        visible_height = min(original_image.shape[2] - y, sampled_image.shape[2])
        return visible_width, visible_height

    def adjust_mask(self, mask, visible_width, visible_height):
        # 调整蒙版大小，使其匹配可见区域的宽度和高度
        mask = mask[:, :, :visible_height, :visible_width]
        # 生成反蒙版，即1减去蒙版
        inverse_mask = 1 - mask
        return mask, inverse_mask

    def compute_composite_portion(self, original_image, sampled_image, mask, inverse_mask, visible_width,
                                  visible_height, x, y):
        # 计算源图像的可见部分，应用蒙版
        sampled_image_portion = mask * sampled_image[:, :, :visible_height, :visible_width]
        # 计算目标图像的可见部分，应用反蒙版
        original_image_portion = inverse_mask * original_image[:, :, y:y + visible_height, x:x + visible_width]
        # 返回合成的图像部分
        return sampled_image_portion + original_image_portion

    def repeat_last_image(self, image, repeat_count):
        return torch.cat((image, image[-1].unsqueeze(0).repeat(repeat_count, 1, 1, 1)), dim=0)


class NIX_RotateCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "rotate_image": ("IMAGE",),
                              "original_image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate_crop"
    CATEGORY = "NIX"

    def rotate_crop(self, rotate_image, original_image):
        r_height, r_width = rotate_image.shape[1], rotate_image.shape[2]
        o_height, o_width = original_image.shape[1], original_image.shape[2]

        x = int((r_width - o_width) / 2)
        y = int((r_height - o_height) / 2)
        to_x = o_width + x
        to_y = o_height + y
        img = rotate_image[:,y:to_y, x:to_x, :]
        return (img,)


class NIX_ImageUpscaleProportionally:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE",),
                    "side_length": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                    "side": (["Longest", "Shortest", "Width", "Height"], {"default": "Longest"}),
                    "eight_multiples": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled", "default": True})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_upscale_proportionally"
    CATEGORY = "NIX"

    # 放大图像的方法
    def image_upscale_proportionally(self, image, side_length, side, eight_multiples=True):

        height_B, width_B = float(image.shape[1]), float(image.shape[2])

        width = width_B  # 目标宽度
        height = height_B  # 目标高度

        # 根据选定的边来确定宽和高
        def determineSide(_side: str) -> tuple[float, float]:
            if _side == "Width":
                height_ratio = height_B / width_B
                width = side_length
                height = height_ratio * width
            elif _side == "Height":
                width_ratio = width_B / height_B
                height = side_length
                width = width_ratio * height
            return width, height

        # 根据选择的边进行条件判断并调整尺寸
        if side == "Longest":
            if width > height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        elif side == "Shortest":
            if width < height:
                width, height = determineSide("Width")
            else:
                width, height = determineSide("Height")
        else:
            width, height = determineSide(side)

        if eight_multiples:
            width = round(width / 8) * 8
            height = round(height / 8) * 8
        else:
            width = round(width)
            height = round(height)

        # 调用通用缩放方法进行图像缩放操作
        def upscale_image(image, width, height):
            # 调整图像维度，将其移动到所需维度
            image = image.movedim(-1, 1)
            # 使用指定算法（lanczos）进行上采样并禁用特定选项
            image = comfy.utils.common_upscale(image, width, height, "lanczos", "disabled")
            # 恢复图像维度
            return image.movedim(1, -1)

        new_image = upscale_image(image, width, height)
        return (new_image,)  # 返回新的图像


class NIX_SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()  # 获取输出目录
        self.type = "output"  # 节点类型为输出
        self.prefix_append = ""  # 文件名前缀附加字符串
        self.compress_level = 4  # PNG压缩级别（只影响体积，不影响画质）

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
            "optional": {
                "output_path": ("STRING", {"default": ""}),
                # 将 save_workflow 改为 image_format（联动开关）：
                # 开启=png（写入工作流元数据），关闭=jpg（不写入工作流元数据）
                "image_format": ("BOOLEAN", {"label_on": "png", "label_off": "jpg", "default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "NIX"

    def _sanitize_filename_prefix(self, name: str) -> str:
        # 去除路径分隔符和不合法字符，避免跨平台保存报错
        # 保留常见安全字符：中文、英文、数字、空格、._-()
        name = name.strip()
        name = name.replace(os.sep, "_")
        name = re.sub(r'[^\w\u4e00-\u9fa5 \.\-\(\)_]+', "_", name)
        return name if name else "output"

    def _prepare_folder_and_base(self, output_path: str, filename_prefix: str):
        if output_path:
            full_output_folder = output_path
            subfolder = ""
        else:
            full_output_folder = self.output_dir
            subfolder = ""

        os.makedirs(full_output_folder, exist_ok=True)

        # 去掉用户可能手动带的扩展名，使用我们后续统一的扩展名
        stem, _ext = os.path.splitext(filename_prefix)
        if stem == "":
            stem = filename_prefix
        filename_stem = os.path.join(full_output_folder, stem)

        return full_output_folder, filename_stem, subfolder

    def _next_available_path(self, filename_stem: str, ext: str, used_names: set) -> str:
        candidate = f"{filename_stem}{ext}"
        if (candidate not in used_names) and (not os.path.exists(candidate)):
            return candidate

        suffix = 2
        while True:
            candidate = f"{filename_stem}_{suffix}{ext}"
            if (candidate not in used_names) and (not os.path.exists(candidate)):
                return candidate
            suffix += 1

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None,
                    output_path="", image_format=True):
        # 处理前缀与格式
        filename_prefix = self._sanitize_filename_prefix(filename_prefix + self.prefix_append)
        fmt_is_png = bool(image_format)
        ext = ".png" if fmt_is_png else ".jpg"

        enable_preview = output_path == ""

        # 路径与基名（不含扩展名）
        full_output_folder, filename_stem, subfolder = self._prepare_folder_and_base(output_path, filename_prefix)

        results = []
        used_names = set()  # 本批次已占用路径，避免同一批次中重复命名

        for _, image in enumerate(images):
            # 张量/数组 -> PIL.Image
            img_array = (255. * image.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # 构造目标路径
            candidate_path = self._next_available_path(filename_stem, ext, used_names)
            used_names.add(candidate_path)

            if fmt_is_png:
                # PNG：写入工作流元数据（除非全局禁用）
                metadata = PngInfo()
                if not args.disable_metadata:
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
                    if extra_pnginfo:
                        for key, value in extra_pnginfo.items():
                            metadata.add_text(str(key), json.dumps(value, ensure_ascii=False))

                img.save(candidate_path, pnginfo=metadata, compress_level=self.compress_level)
            else:
                # JPG：最高质量保存，不写入 PNG 文本元数据
                if img.mode in ("RGBA", "LA"):
                    img = img.convert("RGB")
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(
                    candidate_path,
                    quality=100,       # 最高质量
                    subsampling=0,     # 禁用色度子采样
                    optimize=True      # Huffman 表优化
                )

            file = os.path.basename(candidate_path)
            absolute_path = os.path.abspath(candidate_path)

            result = {
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
                "path": absolute_path
            }

            if enable_preview:
                results.append(result)

        return {"ui": {"images": results if enable_preview else []}}


class NIX_ImageTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", {"default": 2, "min": 1, "max": 256, "step": 1, }),
                "overlap_x": ("INT", {"default": 0, "min": 0, "step": 1, }),
                "overlap_y": ("INT", {"default": 0, "min": 0, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "tile_width", "tile_height", "overlap_x", "overlap_y",)
    FUNCTION = "execute"
    CATEGORY = "NIX"

    def execute(self, image, rows, cols, overlap_x, overlap_y):
        # 获取输入图像的高度和宽度
        h, w = image.shape[1:3]

        # 计算瓦片大小（包含重叠部分）
        tile_h = math.ceil(h / rows) + overlap_y
        tile_w = math.ceil(w / cols) + overlap_x

        tiles = []
        for i in range(rows):
            for j in range(cols):
                # 计算当前瓦片的中心坐标
                center_y = (i + 0.5) * (h / rows)
                center_x = (j + 0.5) * (w / cols)

                # 计算瓦片的起始和结束坐标，确保不超出图像边界
                y1 = max(0, int(center_y - tile_h / 2))
                x1 = max(0, int(center_x - tile_w / 2))
                y2 = min(h, y1 + tile_h)
                x2 = min(w, x1 + tile_w)

                # 如果瓦片触及图像边缘，调整起始坐标以保证瓦片大小一致
                if y2 - y1 < tile_h:
                    y1 = max(0, y2 - tile_h)
                if x2 - x1 < tile_w:
                    x1 = max(0, x2 - tile_w)

                # 从原图中裁剪出瓦片
                tile = image[:, y1:y2, x1:x2, :]
                tiles.append(tile)

        # 将所有瓦片在第一个维度（批次维度）上拼接
        tiles = torch.cat(tiles, dim=0)

        # 返回结果：瓦片图像，瓦片宽度，瓦片高度，x方向重叠，y方向重叠
        return (tiles, tile_w, tile_h, overlap_x, overlap_y,)


class NIX_ImageInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ("STRING", {"default": "o3-2025-04-16"}),
                "base_url": ("STRING", {"default": "https://api.openai-proxy.org/v1"}),
                "api_key": ("STRING", {"default": ""}),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_word",)
    FUNCTION = "image_inference"
    CATEGORY = "NIX"

    def image_inference(self, image, model_name, base_url, api_key, user_prompt):
        try:
            import io, base64
            from openai import OpenAI
        except Exception:
            return ("未安装 openai 库，请先 `pip install openai>=1.0.0`。",)

        if image is None or image.shape[0] == 0:
            return ("未提供图像输入。",)

        # 取第一张图并编码为 PNG base64
        img0 = image[0]
        arr = (img0.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            pil = Image.fromarray(arr)
        else:
            pil = Image.fromarray(arr.squeeze(), mode="L").convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        client = OpenAI(api_key=api_key, base_url=(base_url or None))
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt or ""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        }]
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages)
            text = resp.choices[0].message.content
            return (text or "",)
        except Exception as ex:
            return (f"调用 OpenAI 接口失败: {ex}",)


class NIX_MaskNull:
    @classmethod
    def INPUT_TYPES(s):
        # 定义节点的输入类型
        return {
            "required": {
                "mask": ("MASK",),  # 输入图像
            }
        }

    # 定义节点的输出类型和名称
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)

    # 定义节点的主要功能和类别
    FUNCTION = "is_empty_mask"
    CATEGORY = "NIX"

    def is_empty_mask(self, mask):
        return (torch.all(mask == 0).item(),)


class NIX_StringMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "serial_number": ("INT", {"default": 0, "min": 0, "step": 1}),
                "figure": ("INT", {"default": 5, "min": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "string_match"
    CATEGORY = "NIX"

    def string_match(self, serial_number, figure):
        return (str(serial_number).zfill(figure),)


class NIX_XYGridMapper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x_min": ("FLOAT", {"default": 0.1, "min": 0, "max": 5, "step": 0.05}),
                "x_max": ("FLOAT", {"default": 1, "min": 0, "max": 5, "step": 0.05}),
                "x_step": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 5, "step": 0.05}),
                "y_min": ("FLOAT", {"default": 0.1, "min": 0, "max": 5000, "step": 0.05}),
                "y_max": ("FLOAT", {"default": 1, "min": 0, "max": 500000, "step": 0.05}),
                "y_step": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 5000, "step": 0.05}),
                "index": ("INT", {"default": 1, "min": 1, "max": 2000}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("x_value", "y_value", "name",)
    FUNCTION = "map_index_to_xy"
    CATEGORY = "NIX"

    def map_index_to_xy(self, x_min, x_max, x_step, y_min, y_max, y_step, index):
        x_values = []
        current_x = x_min
        while current_x <= x_max + 1e-7:
            x_values.append(round(current_x, 4))
            current_x += x_step

        y_values = []
        current_y = y_min
        while current_y <= y_max + 1e-7:
            y_values.append(current_y)
            current_y += y_step

        x_count = len(x_values)
        y_count = len(y_values)

        if index < 1 or index > x_count * y_count:
            x_value = x_values[0]
            y_value = y_values[0]
            name = f"{y_value}_{x_value}".replace('.', '')
            return (x_value, int(y_value), name)

        index -= 1
        x_index = index % x_count
        y_index = index // x_count

        x_value = x_values[x_index]
        y_value = y_values[y_index]
        name = f"{y_value}_{x_value}".replace('.', '')

        return (x_value, int(y_value), name)


class NIX_StringMatcher:
    @classmethod
    def INPUT_TYPES(cls):
        # 保持原有接口与默认值
        required = {
            "string_1": ("STRING", {"default": ""}),
            "string_2": ("STRING", {"default": ""}),
            "judge_string": ("STRING", {"default": ""}),
        }
        optional = {
            "inputcount": ("INT", {"default": 2, "min": 2, "max": 32, "step": 1}),
        }
        for i in range(3, 33):
            optional[f"string_{i}"] = ("STRING", {"default": ""})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "string_matcher"
    CATEGORY = "NIX"

    def string_matcher(self, string_1, string_2, judge_string, **kwargs):
        # 统一规范化
        def norm(x):
            return "" if x is None else str(x)

        judge = norm(judge_string)

        # 限定 inputcount 范围 [2, 32]
        v = kwargs.get("inputcount", 2)
        try:
            inputcount = int(v)
        except Exception:
            inputcount = 2
        if inputcount < 2:
            inputcount = 2
        elif inputcount > 32:
            inputcount = 32

        # 构建索引->值的映射，仅包含已传入（或固定存在的 1、2）的项，按编号从小到大进行匹配
        values = {1: norm(string_1), 2: norm(string_2)}
        if inputcount > 2:
            for i in range(3, inputcount + 1):
                key = f"string_{i}"
                if key in kwargs:  # 保持与原逻辑一致：仅匹配传入参数范围内的项
                    values[i] = norm(kwargs.get(key, ""))

        # 找到最小编号的精确匹配
        for idx in range(1, inputcount + 1):
            if idx in values and values[idx] == judge:
                return (idx,)

        return (-1,)


class NIX_SwitchAnything:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything_1": (ANY_T, {}),
                "index": ("INT", {"default": 1, "min": 1}),
            }
        }

    RETURN_TYPES = (ANY_T,)
    RETURN_NAMES = ("value",)
    FUNCTION = "switch_anything"
    CATEGORY = "NIX"

    def switch_anything(self, anything_1, index=1, **kwargs):
        # 收集所有 anything_* 输入：1 -> anything_1, 2 -> anything_2, ...
        values = {1: anything_1}
        for k, v in kwargs.items():
            m = re.match(r"^anything_(\d+)$", k)
            if m:
                n = int(m.group(1))
                values[n] = v

        # 规范化 index，最小为 1
        try:
            idx = int(index)
        except Exception:
            idx = 1
        if idx < 1:
            idx = 1

        # 选择并返回；若目标未提供/未连接，则返回 None
        selected = values.get(idx, None)
        return (selected,)




NODE_CLASS_MAPPINGS = {
    "NIX_PathLoading": NIX_PathLoading,
    "NIX_RotateImage": NIX_RotateImage,
    "NIX_MaskCrop": NIX_MaskCrop,
    "NIX_ImageComposite": NIX_ImageComposite,
    "NIX_RotateCrop": NIX_RotateCrop,
    "NIX_ImageUpscaleProportionally": NIX_ImageUpscaleProportionally,
    "NIX_SaveImage": NIX_SaveImage,
    "NIX_ImageTile": NIX_ImageTile,
    "NIX_ImageInference": NIX_ImageInference,
    "NIX_MaskNull": NIX_MaskNull,
    "NIX_StringMatch": NIX_StringMatch,
    "NIX_XYGridMapper": NIX_XYGridMapper,
    "NIX_StringMatcher": NIX_StringMatcher,
    "NIX_SwitchAnything": NIX_SwitchAnything,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NIX_PathLoading": "NIX_PathLoading",
    "NIX_RotateImage": "NIX_RotateImage",
    "NIX_MaskCrop": "NIX_MaskCrop",
    "NIX_ImageComposite": "NIX_ImageComposite",
    "NIX_RotateCrop": "NIX_RotateCrop",
    "NIX_ImageUpscaleProportionally": "NIX_ImageUpscaleProportionally",
    "NIX_SaveImage": "NIX_SaveImage",
    "NIX_ImageTile": "NIX_ImageTile",
    "NIX_ImageInference": "NIX_ImageInference",
    "NIX_MaskNull": "NIX_MaskNull",
    "NIX_StringMatch": "NIX_StringMatch",
    "NIX_XYGridMapper": "NIX_XYGridMapper",
    "NIX_StringMatcher": "NIX_StringMatcher",
    "NIX_SwitchAnything": "NIX_SwitchAnything",
}
