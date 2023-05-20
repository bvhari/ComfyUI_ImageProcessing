import torch
from kornia.filters import bilateral_blur, unsharp_mask


class BilateralFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE", ),
                             "kernel_size": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                             "sigma_color": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "sigma_space": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.01}),
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
                             "sigma": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "ImageProcessing"

    def sharpen(self, images, kernel_size, sigma):
        images = images.movedim(-1, 1).cpu()
        images_transformed = unsharp_mask(images, (kernel_size, kernel_size), (sigma, sigma))
        images_transformed = images_transformed.movedim(1, -1)

        return (images_transformed,)


NODE_CLASS_MAPPINGS = {
    "BilateralFilter": BilateralFilter,
    "UnsharpMask": UnsharpMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BilateralFilter": "Bilateral Filter",
    "UnsharpMask": "Unsharp Mask",
}
