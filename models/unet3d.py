from monai.networks.nets import UNet


class Unet3D(UNet):
    """
    MONAI 3D U-Net configured for binary segmentation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units: int = 2,
        norm: str = "INSTANCE",
        act: str = "RELU",
        **kwargs,
    ):
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
            act=act,
        )

    def forward(self, x):
        return super().forward(x)

