import albumentations as A


class Perturbation:
    # https://github.com/bcmi/ObjectStitch-Image-Composition/blob/main/ldm/data/open_images.py
    def __init__(self,
                 blur_cfg,
                 color_jitter_cfg,
                 horizontal_flip_cfg,
                 rotate_cfg,
                 perspective_cfg):
        self.blur = A.Blur(**blur_cfg)
        self.appearance_trans = A.Compose([
            A.ColorJitter(**color_jitter_cfg)
        ])

        self.geometric_trans = A.Compose([
            A.HorizontalFlip(**horizontal_flip_cfg),
            A.Rotate(**rotate_cfg),
            A.Perspective(**perspective_cfg)
        ])

    def __call__(self, foreground_image, foreground_mask):
        foreground_image = self.appearance_trans(image=foreground_image)["image"]

        transformed = self.geometric_trans(image=foreground_image, mask=foreground_mask)
        trans_foreground_image = transformed['image']
        trans_foreground_mask = transformed['mask']
        return trans_foreground_image, trans_foreground_mask
