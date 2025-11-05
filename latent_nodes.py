import torch

class QwenImageLatentAspectRatio:
    ASPECT_RATIOS = {
         "Qwen 3:4 (1140x1472)": (1140, 1472),
         "Qwen 4:3 (1472x1140)": (1472, 1140),
         "Qwen 2:3 (1056x1548)": (1056, 1584),
         "Qwen 3:2 (1548x1056)": (1584, 1056),
         "Qwen 1:1 (1328x1328)": (1328, 1328),
         "Qwen 16:9 (1664x928)": (1664,928),
         "Qwen 9:16 (928x1664)": (928, 1664),
         "Super 3:4 (1536x2048)": (1536, 2048),
         "super 4:3 (2048x1536)": (2048, 1536),
         "Super 2:3 (1364x2048)": (1364, 2048),
         "Super 3:2 (2048x1364)": (2048, 1364),
         "Super 9:16 (1152x2048)": (1152, 2048),
         "Super 16:9 (2048x1152)": (2048, 1152),
         "2K 3:4 (2560x3416)": (2560, 3416),
         "2K 4:3 (3416x2560)": (3416, 2560),
         "2K 2:3 (2560x3840)": (2560, 3840),
         "2K 3:2 (3840x2560)": (3840, 2560),
         "2K 9:16 (1440x2560)": (1440, 2560),
         "2K 16:9 (2560x1440)": (2560, 1440),
         "Flux 3:4": (896, 1152),
         "Flux 4:3": (1152, 896),
         "Flux 2:3": (832, 1280),
         "Flux 3:2": (1280, 832),
         "Flux 1:1": (1024, 1024),
         "Flux 16:9": (1344, 768),
         "Flux 9:16": (768, 1344),
         "16:10(1536x960)": (1536, 960),
         "10:16(960x1536)": (960, 1536),
         "21:9(2560x1080)": (2560, 1080),
         "9:21(1080x2560)": (1080, 2560),
         "iPhone 13(1170x2532)": (1170, 2532),
         "安卓主流(1080x2400)": (1080, 2400),
         "三星S22 Ultra(1440x3088)": (1440, 3088),
         "iPad mini(1488x2266)": (1488, 2266),
         "iPad Pro 11(1668x2388)": (1668, 2388),
         "安卓平板(1600x2560)": (1600, 2560),
         "Surface Pro 9(1440x2160)": (1440, 2160),
         "FHD(1920x1080)": (1920, 1080),
         "2K(2560x1440)": (2560, 1440),
         "4K(3840x2160)": (3840, 2160),
         "带鱼屏(3440x1440)": (3440, 1440),
         "MacBook Pro 16(3072x1920)": (3072, 1920),
    }
    
    @classmethod
    def INPUT_TYPES(s):
        ratio_list = ["Custom 自定义"] + list(s.ASPECT_RATIOS.keys())
        return {
            "required": {
                "aspect_ratio": (ratio_list, {"default": "1:1 方形 (1328x1328)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1328, "min": 256, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 1328, "min": 256, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "latent/Qwen_Image_Toolkit"

    def generate_latent(self, aspect_ratio, batch_size, custom_width=1328, custom_height=1328):
        # [修正] 判断条件恢复为中文
        if aspect_ratio == "Custom 自定义":
            width = custom_width
            height = custom_height
            # Silently ensure dimensions are multiples of 8
            if width % 8 != 0: width = round(width / 8) * 8
            if height % 8 != 0: height = round(height / 8) * 8
        else:
            width, height = self.ASPECT_RATIOS[aspect_ratio]
        
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent},)
