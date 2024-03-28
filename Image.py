class Image:
    def __init__(self):
        self.image_in_roi = None
        self.image_out_roi = None
        self.original_image = None
        self.original_image_data = None
        self.image_data = None
        self.ft_data = None
        self.shifted_ft_data = None
        self.ft_magnitude = None
        self.ft_phase = None
        self.ft_real = None
        self.ft_imaginary = None

    def attr(self, mode):
        return getattr(self, '_'.join(mode.split()).lower())