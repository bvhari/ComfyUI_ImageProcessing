import torch
from kornia.filters import bilateral_blur, unsharp_mask
from kornia.enhance import adjust_saturation, adjust_hue, adjust_brightness, adjust_gamma, adjust_sigmoid


class BilateralFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "kernel_size": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                             "sigma_color": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "sigma_space": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "bilateral_filter"

    CATEGORY = "ImageProcessing"

    def bilateral_filter(self, images, kernel_size, sigma_color, sigma_space):
        images = images.movedim(-1, 1).cpu()
        images_transformed = bilateral_blur(images, (kernel_size, kernel_size), sigma_color, (sigma_space, sigma_space), color_distance_type="l2")
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)


class UnsharpMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "kernel_size": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                             "sigma": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "ImageProcessing"

    def sharpen(self, images, kernel_size, sigma):
        images = images.movedim(-1, 1).cpu()
        images_transformed = unsharp_mask(images, (kernel_size, kernel_size), (sigma, sigma))
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)


class Hue:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "factor": ("FLOAT", {"default": 0.0, "min": -3.141516, "max": 3.141516, "step": 0.001}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hue"

    CATEGORY = "ImageProcessing"

    def hue(self, images, factor):
        images = images.movedim(-1, 1).cpu()
        images_transformed = adjust_hue(images, factor)
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)


class Saturation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "saturation"

    CATEGORY = "ImageProcessing"

    def saturation(self, images, factor):
        images = images.movedim(-1, 1).cpu()
        images_transformed = adjust_saturation(images, factor)
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)


class Brightness:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "brightness"

    CATEGORY = "ImageProcessing"

    def brightness(self, images, factor):
        images = images.movedim(-1, 1).cpu()
        images_transformed = adjust_brightness(images, factor)
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)
    

class Gamma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "gamma_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gamma"

    CATEGORY = "ImageProcessing"

    def gamma(self, images, gamma_value):
        images = images.movedim(-1, 1).cpu()
        images_transformed = adjust_gamma(images, gamma_value)
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)


class SigmoidCorrection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "cutoff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "gain": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sigmoid"

    CATEGORY = "ImageProcessing"

    def sigmoid(self, images, cutoff, gain):
        images = images.movedim(-1, 1).cpu()
        images_transformed = adjust_sigmoid(images, cutoff, gain)
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)


NODE_CLASS_MAPPINGS = {
    "BilateralFilter": BilateralFilter,
    "UnsharpMask": UnsharpMask,
    "Hue": Hue,
    "Saturation": Saturation,
    "Brightness": Brightness,
    "Gamma": Gamma,
    "SigmoidCorrection": SigmoidCorrection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BilateralFilter": "Bilateral Filter",
    "UnsharpMask": "Unsharp Mask",
    "Hue": "Hue",
    "Saturation": "Saturation",
    "Brightness": "Brightness",
    "Gamma": "Gamma",
    "SigmoidCorrection": "Sigmoid Correction",
}
