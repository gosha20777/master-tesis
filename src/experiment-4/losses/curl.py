import torch
import torch.nn as nn
from utils import ImageProcessing
from torch.autograd import Variable
import math
from math import exp
import torch.nn.functional as F


class CURLLoss(nn.Module):

    def __init__(self, ssim_window_size=5, alpha=0.5):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLoss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.cuda() + mu2_sq.cuda() + C1) *
                     (sigma1_sq.cuda() + sigma2_sq.cuda() + C2))
        ssim_map = ssim_map1.cuda() / ssim_map2.cuda()

        v1 = 2.0 * sigma12.cuda() + C2
        v2 = sigma1_sq.cuda() + sigma2_sq.cuda() + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs


    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.shape[2]!=img2.shape[2]:
                img1=img1.transpose(2,3)

        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                       img1.shape, img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d',
                       img1.ndim)

        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self.compute_ssim(img1, img2)

            # Relu normalize (not compliant with original definition)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Simple normalize (not compliant with original definition)
        # TODO: remove support for normalize == True (kept for backward support)
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:-1] * pow2[-1])
        return output

    def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser):
        """Forward function for the CURL loss

        :param predicted_img_batch_high_res: 
        :param predicted_img_batch_high_res_rgb: 
        :param target_img_batch: Tensor of shape BxCxWxH
        :returns: value of loss function
        :rtype: float

        """
        num_images = target_img_batch.shape[0]
        target_img_batch = target_img_batch

        ssim_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        l1_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        cosine_rgb_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        hsv_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        rgb_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))

        for i in range(0, num_images):

            target_img = target_img_batch[i, :, :, :].cuda()
            predicted_img = predicted_img_batch[i, :, :, :].cuda()

            predicted_img_lab = torch.clamp(
                ImageProcessing.rgb_to_lab(predicted_img.squeeze(0)), 0, 1)
            target_img_lab = torch.clamp(
                ImageProcessing.rgb_to_lab(target_img.squeeze(0)), 0, 1)

            target_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(
                target_img), 0, 1)
            predicted_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(
                predicted_img.squeeze(0)), 0, 1)

            predicted_img_hue = (predicted_img_hsv[0, :, :]*2*math.pi)
            predicted_img_val = predicted_img_hsv[2, :, :]
            predicted_img_sat = predicted_img_hsv[1, :, :]
            target_img_hue = (target_img_hsv[0, :, :]*2*math.pi)
            target_img_val = target_img_hsv[2, :, :]
            target_img_sat = target_img_hsv[1, :, :]

            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)

            ssim_loss_value += (1.0 - ssim_value)

            predicted_img_1 = predicted_img_val * \
                predicted_img_sat*torch.cos(predicted_img_hue)
            predicted_img_2 = predicted_img_val * \
                predicted_img_sat*torch.sin(predicted_img_hue)

            target_img_1 = target_img_val * \
                target_img_sat*torch.cos(target_img_hue)
            target_img_2 = target_img_val * \
                target_img_sat*torch.sin(target_img_hue)

            predicted_img_hsv = torch.stack(
                (predicted_img_1, predicted_img_2, predicted_img_val), 2)
            target_img_hsv = torch.stack((target_img_1, target_img_2, target_img_val), 2)

            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)
            rgb_loss_value += F.l1_loss(predicted_img, target_img)
            hsv_loss_value += F.l1_loss(predicted_img_hsv, target_img_hsv)

            cosine_rgb_loss_value += (1-torch.mean(
                torch.nn.functional.cosine_similarity(predicted_img, target_img, dim=0)))

        l1_loss_value = l1_loss_value/num_images
        rgb_loss_value = rgb_loss_value/num_images
        ssim_loss_value = ssim_loss_value/num_images
        cosine_rgb_loss_value = cosine_rgb_loss_value/num_images
        hsv_loss_value = hsv_loss_value/num_images

        curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
                     hsv_loss_value + 10*ssim_loss_value + 1e-6*gradient_regulariser)/6

        return curl_loss