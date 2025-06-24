import torch
import torch.nn.functional as F
import math


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = math.ceil(height / factor) * factor
    w_bar = math.ceil(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def custom_smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
    assert height == width
    if height < factor:
        resize_h = factor
        resize_w = factor
    else:
        resize_h, resize_w = smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    
    return resize_h, resize_w


def token_res_map_func(max_size, max_token_num, t, p):
    b = (max_size-1) / math.pow(max_token_num-1, p)
    s = int(max_size - b * math.pow(max_token_num-t, p))
    s = min(s, max_size)
    s = max(1, s)
    return s


def resize_imgs(img, size):
    B, C, H, W = img.shape
    new_h, new_w = size
    if H > new_h:
        return F.interpolate(img, size=size, mode='area')
    elif H == new_h:
        assert W == new_w
        return img
    else:
        return F.interpolate(img, size=size, mode='bicubic', align_corners=False)
    

def find_closest_resolution(x, best_res):
    closest = best_res[0]
    for r in best_res:
        if abs(r - x) < abs(closest - x):
            closest = r
    return closest
    

class ResolutionDegradation(object):
    def __init__(self, max_img_size, max_token_num, group_size, causal_num, p, enc_patch_size=None, dec_patch_size=None, min_dec_res=None):
        self.max_img_size = max_img_size
        assert min_dec_res % enc_patch_size ==0 and min_dec_res % dec_patch_size == 0
        self.min_dec_res = min_dec_res  # Constrains the minimum input/output size
        self.max_token_num = max_token_num
        self.group_size = group_size
        assert max_token_num % group_size == 0
        self.max_group_num = self.max_token_num // self.group_size
        self.causal_num = self.group_size if causal_num is None else causal_num
        assert self.causal_num % self.group_size == 0

        self.p = p
        self.enc_patch_size = enc_patch_size
        self.dec_patch_size = dec_patch_size

        self.max_enc_side_patch_num = math.ceil(self.max_img_size / self.enc_patch_size)
        self.max_dec_side_patch_num = math.ceil(self.max_img_size / self.dec_patch_size)
        self.init_mapping()
        assert enc_patch_size <= dec_patch_size
        self.best_token = sorted(list(self.decres2besttoken.values()))
    
    def get_dec_smart_size(self, h_or_w):
        res_size_smart, _ = custom_smart_resize(
            h_or_w, h_or_w, 
            self.dec_patch_size, 
            min_pixels=self.dec_patch_size*self.dec_patch_size, 
            max_pixels=self.max_dec_side_patch_num*self.max_dec_side_patch_num*self.dec_patch_size*self.dec_patch_size
            )
        return res_size_smart
    
    def get_enc_smart_size(self, h_or_w):
        res_size_smart, _ = custom_smart_resize(
            h_or_w, h_or_w, 
            self.enc_patch_size, 
            min_pixels=self.enc_patch_size*self.enc_patch_size, 
            max_pixels=self.max_enc_side_patch_num*self.max_enc_side_patch_num*self.enc_patch_size*self.enc_patch_size,
            )
        return res_size_smart

    def init_mapping(self):
        token_list = list(range(1, self.causal_num+1)) +  [t * self.group_size for t in range(self.causal_num//self.group_size + 1, self.max_group_num+1)]
        token2dres = {}
        token2eres = {}
        token2res = {}

        encres2besttoken = {}
        decres2besttoken = {}
        last_enc_diff = {}
        last_dec_diff = {}

        decres2mintoken = {}
        decres2maxtoken = {}
        for t in token_list:
            res_size = token_res_map_func(self.max_img_size, self.max_token_num, t, p=self.p)
            res_size_smart_dec = self.get_dec_smart_size(res_size)
            res_size_smart_enc = self.get_enc_smart_size(res_size)
            
            token2res[t] = res_size
            token2dres[t] = res_size_smart_dec
            token2eres[t] = res_size_smart_enc

            if abs(res_size-res_size_smart_enc) <= last_enc_diff.get(res_size_smart_enc, self.max_img_size):
                encres2besttoken[res_size_smart_enc] = t
                last_enc_diff[res_size_smart_enc] = abs(res_size-res_size_smart_enc)
            
            if abs(res_size-res_size_smart_dec) <= last_dec_diff.get(res_size_smart_dec, self.max_img_size):
                decres2besttoken[res_size_smart_dec] = t
                last_dec_diff[res_size_smart_dec] = abs(res_size-res_size_smart_dec)

            decres2mintoken[res_size_smart_dec] = min(decres2mintoken.get(res_size_smart_dec, self.max_token_num), t)
            decres2maxtoken[res_size_smart_dec] = max(decres2maxtoken.get(res_size_smart_dec, 1), t)
        self.token2res = token2res
        self.token2dec_res = token2dres
        self.token2enc_res = token2eres
        self.encres2besttoken = encres2besttoken
        self.decres2besttoken = decres2besttoken
        self.decres2mintoken = decres2mintoken
        self.decres2maxtoken = decres2maxtoken

        self.best_res = list(self.decres2besttoken.keys())
        self.token2dec_out_res = {k: max(v, self.min_dec_res) for k, v in self.token2dec_res.items()}
    
    def get_noise_token_idx(self, t, p, correction_training):
        if not correction_training:
            return None
        if t == 1:
            return None
        if torch.rand(1).item() < p:
            if t <= self.causal_num:
                noise_idx_max = min(t, self.causal_num)
            else:
                assert t >= self.group_size + self.causal_num
                assert t % self.group_size == 0
                noise_idx_max = t - self.group_size
            noise_idx = torch.randint(0, noise_idx_max, (1,)).item()
            if noise_idx < self.causal_num:
                return noise_idx
            else:
                noise_idx = noise_idx // self.group_size * self.group_size
                return noise_idx
        else:
            return None
    
    @torch.no_grad()
    def get_input_output_img(self, x, degradation_prob=None):
        x_new = x
        _, _, height, _ = x.shape 
        assert height == find_closest_resolution(height, self.best_res)
        cur_max_token_num = self.decres2besttoken[height]
        t = cur_max_token_num
        tgt_res = height
        dec_res = height

        if degradation_prob is not None and torch.rand(1).item() < degradation_prob:
            assert cur_max_token_num >= self.group_size + self.causal_num
            t = torch.randint(1, cur_max_token_num-self.group_size+1, (1,)).item()
            if t > self.causal_num:
                t = math.ceil(t / self.group_size) * self.group_size
    
            tgt_res = self.token2res[t]
            dec_res = self.token2dec_out_res[t]
            new_size = (tgt_res, tgt_res)
            resized_img = resize_imgs(x, size=new_size)
            x_new = resize_imgs(resized_img, size=(dec_res, dec_res))
        return x_new, t, tgt_res, dec_res, cur_max_token_num
    
    @torch.no_grad()
    def resize_batch_img(self, x, dynamic_resolution_prob=None, max_resolution_prob=None, image_size=None, min_image_size=None, max_image_size=None, adjust_bs_by_resolution=False):
        assert dynamic_resolution_prob >= 0 and dynamic_resolution_prob <= 1
        assert max_resolution_prob >= 0 and max_resolution_prob <= 1
        assert max_resolution_prob + dynamic_resolution_prob <= 1
        assert min_image_size >= self.min_dec_res
        
        cur_p = torch.rand(1).item()

        if dynamic_resolution_prob is not None and min_image_size < max_image_size and cur_p < dynamic_resolution_prob:
            valid_r = None
            for _ in range(100):
                r = torch.randint(min_image_size, max_image_size+1, (1,)).item()
                r = find_closest_resolution(r, self.best_res)
                if r >= min_image_size and r <= max_image_size:
                    valid_r = r
                    break
            
            if valid_r is None:
                raise ValueError(f"{min_image_size}-{max_image_size} from {self.best_res}")
        elif cur_p < dynamic_resolution_prob + max_resolution_prob:
            valid_r = max_image_size
        else:
            valid_r = image_size
        if adjust_bs_by_resolution:
            B = x.shape[0]
            if valid_r > 480 and valid_r <= 512:
                x = x[:int(B*0.5)]
            elif valid_r >= 384 and valid_r <= 480:
                x = x[:int(B*0.6)]

        x = resize_imgs(x, size=(valid_r, valid_r))
        return x


def create_attn_mask_for_x_z(n, m, z_causal=False, x_see_z=False, z_see_x=True, group_size=1, causal_num=None):
    attn_mask = torch.zeros((n + m, n + m), dtype=torch.bool)
    assert m % group_size == 0, "m must be divisible by group_size"
    
    if not x_see_z:
        attn_mask[:n, n:] = True  
    
    if not z_see_x:
        attn_mask[n:, :n] = True  
    
    if z_causal:
        if causal_num is None:
            causal_num = group_size  
        else:
            assert causal_num % group_size == 0, "causal_num must be a multiple of group_size"
            assert causal_num <= m, "causal_num cannot exceed total z tokens m"
        
        for start in range(n, n + m, group_size):
            end = start + group_size
            if start < n + causal_num:
                group_slice = slice(start, end)
                inner_mask = torch.triu(torch.ones((group_size, group_size), dtype=torch.bool), diagonal=1)
                attn_mask[group_slice, group_slice] = inner_mask
                attn_mask[group_slice, end:] = True
            else:
                attn_mask[start:end, end:] = True
    
    return attn_mask


@torch.no_grad()
def average_cosine_similarity(features):
    cosine_similarity = torch.nn.functional.cosine_similarity(
        features.unsqueeze(2), features.unsqueeze(1), dim=-1
    )
    
    batch_size, num_features, _ = features.size()
    mask = torch.eye(num_features, device=features.device).bool()
    cosine_similarity.masked_fill_(mask.unsqueeze(0), 0)

    average_cosine_distance_per_sample = cosine_similarity.sum(dim=(-1, -2)) / (num_features * (num_features - 1))

    average_cosine_distance_batch = average_cosine_distance_per_sample.mean()
    
    return average_cosine_distance_batch


