import albumentations as albu


def get_xview3_noaug(**kwargs):
    train_transform = albu.Compose(
        [
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def get_xview3_augv1(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.OneOf(
                [
                    albu.RandomBrightness(0.1, p=1),
                    albu.RandomContrast(0.1, p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.3,
            ),
            # albu.ShiftScaleRotate(
            #     shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3
            # ),
            # albu.Cutout(p=0.5, max_h_size=80, max_w_size=80, num_holes=5),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def get_xview3_augv2(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.OneOf(
                [
                    # albu.RandomBrightness(0.1, p=1),
                    # albu.RandomContrast(0.1, p=1),
                    albu.RandomBrightnessContrast(0.1, p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.3,
            ),
            albu.CoarseDropout(
                p=0.3,
                min_holes=5,
                max_holes=30,
                max_height=100,
                max_width=100,
                min_height=50,
                min_width=50,
                mask_fill_value=0,
            ),
            albu.Rotate(
                p=0.3,
                limit=(-90, 90),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
            ),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def get_xview3_augv3(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(0.1, p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.3,
            ),
            albu.Rotate(
                p=0.3,
                limit=(-90, 90),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
            ),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def get_xview3_augv3_centercrop64(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(0.1, p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.3,
            ),
            albu.Rotate(
                p=0.3,
                limit=(-90, 90),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
            ),
            albu.CenterCrop(64, 64, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.CenterCrop(64, 64, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def get_xview3_augv3_centercrop92(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(0.1, p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.3,
            ),
            albu.Rotate(
                p=0.3,
                limit=(-90, 90),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
            ),
            albu.CenterCrop(92, 92, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.CenterCrop(64, 64, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def get_xview3_augv4_crop64(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.OneOf(
                [
                    albu.RandomBrightnessContrast(0.1, p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.3,
            ),
            albu.Rotate(
                p=0.3,
                limit=(-90, 90),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
            ),
            albu.CenterCrop(64, 64, p=1.0),
            albu.CoarseDropout(
                p=0.3,
                min_holes=2,
                max_holes=6,
                max_height=6,
                max_width=6,
                min_height=2,
                min_width=2,
                mask_fill_value=0,
            ),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.CenterCrop(64, 64, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def augv5(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.OneOf(
                [
                    # albu.RandomBrightness(0.1, p=1),
                    # albu.RandomContrast(0.1, p=1),
                    albu.RandomBrightnessContrast(0.1, p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.3,
            ),
            albu.CoarseDropout(
                p=0.3,
                min_holes=5,
                max_holes=30,
                max_height=100,
                max_width=100,
                min_height=50,
                min_width=50,
                mask_fill_value=0,
            ),
            # albu.Rotate(
            #     p=0.3,
            #     limit=(-90, 90),
            #     interpolation=0,
            #     border_mode=0,
            #     value=(0, 0, 0),
            #     mask_value=None,
            # ),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform
