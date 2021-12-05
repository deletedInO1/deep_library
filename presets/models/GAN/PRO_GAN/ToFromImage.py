from presets.models.GAN.PRO_GAN.WSConv2D import WSConv2D

class ToImage(WSConv2D): #(ToRGB)
    def __init__(self, in_channels, img_channels):
        super(ToImage, self).__init__(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

    def __str__(self):
        return "ToImage("+str(self.conv.in_channels) +", " + str(self.conv.out_channels)+")"

class FromImage(WSConv2D):
    def __init__(self, img_channels, out_channels):
        super(FromImage, self).__init__(img_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __str__(self):
        return "FromImage("+str(self.conv.in_channels) +", " + str(self.conv.out_channels)+")"
