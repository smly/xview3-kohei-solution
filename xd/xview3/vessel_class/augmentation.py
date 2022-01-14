import albumentations as albu


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


def get_xview3_augv5_centercrop64(**kwargs):
    train_transform = albu.Compose(
        [
            albu.RandomRotate90(p=0.3),
            albu.HorizontalFlip(p=0.3),
            albu.Rotate(
                p=0.3,
                limit=(-90, 90),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=(0, 0, 0),
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


def augv6_crop128(**kwargs):
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
            albu.CenterCrop(128, 128, p=1.0),
            albu.CoarseDropout(
                p=0.1,
                min_holes=2,
                max_holes=6,
                max_height=12,
                max_width=12,
                min_height=4,
                min_width=4,
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
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def augv7(**kwargs):
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
            # albu.CenterCrop(128, 128, p=1.0),
            albu.CoarseDropout(
                p=0.1,
                min_holes=2,
                max_holes=6,
                max_height=12,
                max_width=12,
                min_height=4,
                min_width=4,
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
            # albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def augv7_crop64(**kwargs):
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
            # albu.CenterCrop(128, 128, p=1.0),
            albu.CoarseDropout(
                p=0.1,
                min_holes=2,
                max_holes=6,
                max_height=12,
                max_width=12,
                min_height=4,
                min_width=4,
                mask_fill_value=0,
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


def augv7_crop128(**kwargs):
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
            # albu.CenterCrop(128, 128, p=1.0),
            albu.CoarseDropout(
                p=0.1,
                min_holes=2,
                max_holes=6,
                max_height=12,
                max_width=12,
                min_height=4,
                min_width=4,
                mask_fill_value=0,
            ),
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def augv8_crop128(**kwargs):
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
            albu.CenterCrop(128, 128, p=1.0),
            albu.CoarseDropout(
                p=0.1,
                min_holes=2,
                max_holes=6,
                max_height=12,
                max_width=12,
                min_height=4,
                min_width=4,
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
            # albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def augv9_crop128(**kwargs):
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
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform


def augv10_crop128(**kwargs):
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
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    val_transform = albu.Compose(
        [
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ]
    )
    return train_transform, val_transform
