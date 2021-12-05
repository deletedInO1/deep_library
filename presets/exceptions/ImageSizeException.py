

class ImageSizeException(Exception):
    def __init__(self, img_tensor, expected_size):

        message = "The image tensor has the worng shape. expected: " + str(expected_size)  + ", got:" + str(img_tensor.shape)
        self.img_tensor = img_tensor
        self.excepted_size = expected_size
        super(ImageSizeException, self).__init__(message)