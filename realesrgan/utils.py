import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_AO_OPT_15 = int(os.environ.get('AUTOOPTM_OPT_2', '16'))


def _ao_on(name):
    return os.environ.get(name, '1') != '0'


def _ao_opt_13(out_t, alpha_t, img_mode):
    def bgr(t):
        t = t.data.squeeze(0).float().clamp_(0, 1)
        return t[[2, 1, 0], :, :].permute(1, 2, 0)

    def gray(t):                                    # COLOR_BGR2GRAY
        return t[:, :, 0] * 0.114 + t[:, :, 1] * 0.587 + t[:, :, 2] * 0.299

    out = bgr(out_t)
    if img_mode == 'L':
        out = gray(out)
    elif img_mode == 'RGBA':
        out = torch.cat([out, gray(bgr(alpha_t)).unsqueeze(-1)], dim=-1)
    return out.mul_(255.0).round_().to(torch.uint8).contiguous().cpu().numpy()


_AO_T = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def _ao_opt_14(model):
    from torch import nn
    needed = ('conv_first', 'conv_body', 'body', 'conv_up1', 'conv_up2',
              'conv_hr', 'conv_last', 'lrelu')
    if not all(hasattr(model, a) for a in needed):
        return False                    # not an RRDBNet (the compact nets differ)
    if model.conv_up1.kernel_size != (3, 3) or model.conv_up1.stride != (1, 1):
        return False

    def convert(conv):
        w = conv.weight.detach().float()
        T = torch.tensor(_AO_T, device=w.device, dtype=w.dtype)
        ct = nn.ConvTranspose2d(conv.in_channels, conv.out_channels, 4, 2, 1,
                                bias=conv.bias is not None)
        ct.weight.data.copy_(torch.einsum('ku,oiuv,lv->iokl', T, w, T).to(ct.weight.dtype))
        if conv.bias is not None:
            ct.bias.data.copy_(conv.bias.detach().float().to(ct.bias.dtype))
        return ct.to(device=conv.weight.device, dtype=conv.weight.dtype)

    up1, up2 = convert(model.conv_up1), convert(model.conv_up2)

    def forward(x):
        feat = model.conv_first(x)
        feat = feat + model.conv_body(model.body(feat))
        feat = model.lrelu(up1(feat))
        feat = model.lrelu(up2(feat))
        return model.conv_last(model.lrelu(model.conv_hr(feat)))

    # Prove the rewrite on a small probe before trusting it with real images.
    with torch.no_grad():
        probe = torch.randn(1, model.conv_first.in_channels, 32, 40,
                            device=model.conv_first.weight.device,
                            dtype=model.conv_first.weight.dtype)
        ref = model(probe).float()
        err = (forward(probe).float() - ref).abs().max().item()
        scale = ref.abs().max().item() or 1.0
    if err / scale > 5e-3:
        print('[autooptm] optimized path disagrees by %.3g; keeping the stock path' % err)
        return False
    model.up1, model.up2, model.forward = up1, up2, forward
    return True


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

        if _ao_on('AUTOOPTM_OPT_3'):
            _ao_opt_14(self.model)
        if self.device.type == 'cuda' and _ao_on('AUTOOPTM_OPT_4'):
            torch.backends.cudnn.benchmark = True
        if self.device.type == 'cuda' and _ao_on('AUTOOPTM_OPT_5'):
            self.model = self.model.to(memory_format=torch.channels_last)

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
        if self.img.is_cuda and _ao_on('AUTOOPTM_OPT_5'):
            self.img = self.img.contiguous(memory_format=torch.channels_last)

    def process(self):
        # model inference
        runner = self._ao_opt_16()
        self.output = runner(self.img) if runner is not None else self.model(self.img)

    def _ao_opt_16(self):
        if getattr(self, '_ao_off', False) or self.device.type != 'cuda':
            return None
        if not hasattr(self, '_ao_opt_18'):
            self._ao_opt_18 = {}
            if _ao_on('AUTOOPTM_OPT_6'):
                from basicsr.archs.rrdbnet_arch import RRDB
                if not getattr(RRDB, '_ao_opt_21', False):
                    RRDB.forward = torch.compile(RRDB.forward)
                    RRDB._ao_opt_21 = True
        if not _ao_on('AUTOOPTM_OPT_7'):
            return self.model
        key = tuple(self.img.shape)
        hit = self._ao_opt_18.get(key)
        if hit is None:
            if len(self._ao_opt_18) >= _AO_OPT_15:
                return self.model
            try:
                opt_19 = torch.empty_like(self.img)
                opt_19.copy_(self.img)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self.model(opt_19)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    opt_20 = self.model(opt_19)
            except RuntimeError as error:
                print('[autooptm] optimized path unavailable:', error)
                self._ao_off = True
                return self.model
            hit = (g, opt_19, opt_20)
            self._ao_opt_18[key] = hit

        def replay(x, _hit=hit):
            g, opt_19, opt_20 = _hit
            opt_19.copy_(x)
            g.replay()
            return opt_20.clone()

        return replay


    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _ao_opt_17 = (self.device.type == 'cuda' and max_range == 255
                   and alpha_upsampler == 'realesrgan'
                   and _ao_on('AUTOOPTM_OPT_8'))

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img_t, output_alpha_t = output_img, None
        if not _ao_opt_17:
            output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
            if img_mode == 'L':
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha_t = output_alpha
                if not _ao_opt_17:
                    output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                    output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            if not _ao_opt_17:
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
                output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if _ao_opt_17:
            output = _ao_opt_13(output_img_t, output_alpha_t, img_mode)
        elif max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode


class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class IOConsumer(threading.Thread):

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')
