import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import numpy as np
import math 
from typing import Tuple
from matplotlib import pyplot as plt
#from reshape_distributions import ReshapedDistribution

#num_class = 19
num_class = 150
#num_class = 171
#num_class = 2


def make_model_diagrams(outputs, labels,  n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    softmaxes = torch.nn.functional.softmax(outputs, 1)
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)
    
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
    
    plt.figure(0, figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    bin_corrects = np.nan_to_num(np.array([bin_correct  for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

    ece = _calculate_ece(outputs, labels)

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig('reliability_diagram.png')
    plt.show()
    return ece

def _calculate_ece(logits, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()
    
def initialization(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        
    
        
class ReshapedDistribution(td.Distribution):
    def __init__(self, base_distribution: td.Distribution, new_event_shape: Tuple[int, ...]):
        super().__init__(batch_shape=base_distribution.batch_shape, event_shape=new_event_shape, validate_args=False)
        
        self.base_distribution = base_distribution
#        print(td.Distribution.batch_shape)
#        print(self.base_distribution.arg_constraints())
        
        self.new_shape = base_distribution.batch_shape + new_event_shape
#        self.arg_constraints = self.base_distribution.arg_constraints()

    @property
    def support(self):
        return self.base_distribution.support

#    @property
#    def arg_constraints(self):
#        return self.base_distribution.arg_constraints()
#        return td.Distribution.arg_constraints()

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(sample_shape + self.new_shape)

    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()
############################################################################################

class Temperature_Scaling(nn.Module):
    def __init__(self):
        super(Temperature_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))

    def weights_init(self):
        self.temperature_single.data.fill_(1)

    def forward(self, logits):
        temperature = self.temperature_single.expand(logits.size()).cuda()
        return logits / temperature


class Vector_Scaling(nn.Module):
    def __init__(self):
        super(Vector_Scaling, self).__init__()
        self.vector_parameters = nn.Parameter(torch.ones(1, num_class, 1, 1))
        self.vector_offset = nn.Parameter(torch.zeros(1, num_class, 1, 1))

    def weights_init(self):
#        pass
        self.vector_offset.data.fill_(0)
        self.vector_parameters.data.fill_(1)

    def forward(self, logits):
        return logits * self.vector_parameters.cuda() + self.vector_offset.cuda()
        
class Stochastic_Spatial_Scaling(nn.Module):
    def __init__(self):
        super(Stochastic_Spatial_Scaling, self).__init__()

        conv_fn = nn.Conv2d
        self.rank = 10
        self.num_classes = num_class
        self.epsilon = 1e-5
        self.diagonal = False  # whether to use only the diagonal (independent normals)
#        self.mean_l = conv_fn(num_class, num_class, kernel_size=(1, ) * 2)
#        self.log_cov_diag_l = conv_fn(num_class, num_class, kernel_size=(1, ) * 2)
#        self.cov_factor_l = conv_fn(num_class, num_class * 10, kernel_size=(1, ) * 2)
        self.conv_logits = conv_fn(num_class, num_class, kernel_size=(1, ) * 2)

    def weights_init(self):
#        initialization(self.mean_l)
#        initialization(self.log_cov_diag_l)
#        initialization(self.cov_factor_l)       
        
        initialization(self.conv_logits)
        
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean
                
    def forward(self, logits):
#        logits = F.relu(super().forward(image, **kwargs)[0])
#        logits = logits.permute(0,2,3,1)
#        print(logits.shape)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]


        mean = self.conv_logits(logits)
        cov_diag = (mean*1e-5).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))        


#        mean = self.mean_l(logits)
#        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
#        mean = mean.view((batch_size, -1))
#        cov_diag = cov_diag.view((batch_size, -1))
        

#        cov_factor = self.cov_factor_l(logits)
#        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
#        cov_factor = cov_factor.flatten(2, 3)
#        cov_factor = cov_factor.transpose(1, 2)
#
#        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
##        mask = kwargs['sampling_mask']
##        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
##        cov_factor = cov_factor * mask.unsqueeze(-1)
##        cov_diag = cov_diag * mask + self.epsilon
#        
#        cov_factor = cov_factor 
#        cov_diag = cov_diag + self.epsilon
#
#        if self.diagonal:
#            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
#        else:
#            try:
#                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
#            except:
#                print('Covariance became not invertible using independent normals for this batch!')
#                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
#                

        base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        distribution = ReshapedDistribution(base_distribution, event_shape)

#        shape = (batch_size,) + event_shape
#        logit_mean = mean.view(shape)
  
        
#        cov_diag_view = cov_diag.view(shape).detach()
#        cov_factor_view = cov_factor.transpose(2, 1).view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()

#        output_dict = {'logit_mean': logit_mean.detach(),
#                       'cov_diag': cov_diag_view,
#                       'cov_factor': cov_factor_view,
#                       'distribution': distribution}
                       
#        logit_sample = self.fixed_re_parametrization_trick(distribution.rsample(sample_shape=logit_mean.shape[1]), num_samples=10)
        num_samples=2
        samples = distribution.rsample((num_samples // 2,)).cpu()
        mean = distribution.mean.unsqueeze(0).cpu()
        samples = samples - mean
        logit_samples = torch.cat([samples, -samples]) + mean
        logit_mean = logit_samples.mean(dim=0).cuda()

        return logit_mean
        
#class Logistic_Scaling(nn.Module):
#    def __init__(self):
#        super(Logistic_Scaling, self).__init__()
##        self.logistic_C = nn.Parameter(torch.ones(1, num_class, 1, 1))
##        self.logistic_R = nn.Parameter(torch.ones(1, num_class, 1, 1))
#        self.logistic_I = nn.Parameter(torch.ones(1, num_class, 1, 1))
#        self.logistic_K = nn.Parameter(torch.ones(1, num_class, 1, 1))
#
#    def weights_init(self):
##        pass
##         self.logistic_C.data.fill_(0)
##         self.logistic_R.data.fill_(1)
#         self.logistic_I.data.fill_(0)
#         self.logistic_K.data.fill_(1)
#
#    def forward(self, logits, args):
##        return self.logistic_C.cuda(args.gpu) + self.logistic_R.cuda(args.gpu) / (1+ torch.exp(self.logistic_K.cuda(args.gpu) * (probs + self.logistic_I.cuda(args.gpu)) ))
##        softmax = torch.nn.Softmax(dim=1)
##        probs = softmax(logits)
##        return 1/(1 + torch.exp(-self.logistic_K*logits + self.logistic_I))    
#        return   torch.exp(self.logistic_K*logits + self.logistic_I) / torch.sum(torch.exp(self.logistic_K*logits + self.logistic_I), dim=1, keepdim=True)

class Dirichlet_Scaling(nn.Module):
    def __init__(self):
        super(Dirichlet_Scaling, self).__init__()
        self.dirichlet_linear = nn.Linear(num_class, num_class)

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))
#        pass
    def forward(self, logits):
    
#        logits = logits.permute(0,2,3,1).view(-1, num_class)
#        gt = gt.view(-1)

        logits = logits.permute(0,2,3,1)
#        gt = gt.view(-1)
        
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)
        ln_probs = torch.log(probs+1e-10)
#        print(ln_probs.shape)

        return self.dirichlet_linear(ln_probs).permute(0,3,1,2)   
#        return self.dirichlet_linear(ln_probs), gt
#        return self.dirichlet_linear(ln_probs)
        
        
class Meta_Scaling(nn.Module):
    def __init__(self):
        super(Meta_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))
        self.alpha = 0.05
#        self.threshold_linear = nn.Linear(1000, 1)

    def weights_init(self):
#        self.threshold_linear.weight.data.copy_(torch.zeros(self.threshold_linear.weight.shape))
        self.temperature_single.data.fill_(1)
        
    def forward(self, logits, gt, threshold):

        logits = logits.permute(0,2,3,1).view(-1, num_class)
        gt = gt.view(-1)
    
        if self.training:
            neg_ind = torch.argmax(logits, axis=1) == gt
            
            xs_pos, ys_pos = logits[~neg_ind], gt[~neg_ind]
            xs_neg, ys_neg = logits[neg_ind], gt[neg_ind]
            
            start = np.random.randint(int(xs_neg.shape[0]*1/3))+1
            x2 = torch.cat((xs_pos, xs_neg[start:int(xs_neg.shape[0]/2)+start]), 0)
            y2 = torch.cat((ys_pos, ys_neg[start:int(xs_neg.shape[0]/2)+start]), 0)
            
            softmax = torch.nn.Softmax(dim=-1)
            p = softmax(x2)
            scores_x2 = torch.sum(-p*torch.log(p), dim=-1)
        
            cond_ind = scores_x2 < threshold
            cal_logits, cal_gt = x2[cond_ind], y2[cond_ind]
        
            temperature = self.temperature_single.expand(cal_logits.size())
            cal_logits = cal_logits / temperature
            
        else:
            x2 = logits
            y2 = gt
        
            softmax = torch.nn.Softmax(dim=-1)
            p = softmax(x2)
            scores_x2 = torch.sum(-p*torch.log(p), dim=-1)
        
            cond_ind = scores_x2 < threshold
            scaled_logits, scaled_gt = x2[cond_ind], y2[cond_ind]
            inference_logits, inference_gt = x2[~cond_ind], y2[~cond_ind]
        
            temperature = self.temperature_single.expand(scaled_logits.size())
#            print(self.temperature_single)
            scaled_logits = scaled_logits / temperature
            
#            p = torch.tensor(1/num_class, dtype=torch.float).cuda()           
#            inference_logits = torch.log(p)*torch.ones_like(inference_logits)

            inference_logits = torch.ones_like(inference_logits)
#            inference_logits = inference_logits/100
            
            cal_logits = torch.cat((scaled_logits, inference_logits), 0)
            cal_gt = torch.cat((scaled_gt, inference_gt), 0)


#        loss = nn.CrossEntropyLoss(ignore_index=255)(cal_logits, cal_gt)
#        return loss

        return cal_logits, cal_gt


class IBTS_CamVid_With_Image(nn.Module):
    def __init__(self):
        super(IBTS_CamVid_With_Image, self).__init__()
        self.temperature_level_2_conv1 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

    def forward(self, logits, image, args):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda(args.gpu)
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda(args.gpu)
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda(args.gpu)
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda(args.gpu)
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_num_class = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_num_class * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + (torch.ones(1)).cuda(args.gpu)
        temp_param = self.temperature_level_2_param_img(logits)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(torch.mean(temperature) + torch.ones(1).cuda(args.gpu)) + sigma
        return logits / temperature

class LTS_CamVid_With_Image(nn.Module):
    def __init__(self):
        super(LTS_CamVid_With_Image, self).__init__()
        self.temperature_level_2_conv1 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param_img = nn.Conv2d(num_class, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)

    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

#        torch.nn.init.constant_(self.temperature_level_2_conv1.weight.data, 0.001)
#        torch.nn.init.constant_(self.temperature_level_2_conv1.bias.data, 0.001)
#        torch.nn.init.constant_(self.temperature_level_2_conv2.weight.data, 0.001)
#        torch.nn.init.constant_(self.temperature_level_2_conv2.bias.data, 0.001)
#        torch.nn.init.constant_(self.temperature_level_2_conv3.weight.data, 0.001)
#        torch.nn.init.constant_(self.temperature_level_2_conv3.bias.data, 0.001)
#        torch.nn.init.constant_(self.temperature_level_2_conv4.weight.data, 0.001)
#        torch.nn.init.constant_(self.temperature_level_2_conv4.bias.data, 0.001)
#        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
#        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
#        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
#        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
#        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
#        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
#        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
#        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
#        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
#        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)


    def forward(self, logits, image):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda()
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda()
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda()
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda()
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_num_class = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_num_class * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + torch.ones(1).cuda()
        temp_param = self.temperature_level_2_param_img(logits)
        temperature = temp_1 * torch.sigmoid(temp_param) + temp_2 * (1.0 - torch.sigmoid(temp_param))
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda()) + sigma
        temperature = temperature.repeat(1, num_class, 1, 1)
        return logits / temperature



################
class Binary_Classifier(nn.Module):
    def __init__(self):
        super(Binary_Classifier, self).__init__()
        self.dirichlet_linear = nn.Linear(num_class, num_class)
        self.binary_linear = nn.Linear(num_class, 2)
        
        self.bn0 = nn.BatchNorm2d(num_class)
        self.linear_1 = nn.Linear(num_class, num_class*2)
        self.bn1 = nn.BatchNorm2d(num_class*2)
        self.linear_2 = nn.Linear(num_class*2, num_class)
        self.bn2 = nn.BatchNorm2d(num_class)

        self.relu = nn.ReLU()        

#        self.linear_3 = nn.Linear(num_class*2, num_class*2)
#        self.bn3 = nn.BatchNorm2d(num_class*2)
#        self.linear_4 = nn.Linear(num_class*2, num_class)
#        self.bn4 = nn.BatchNorm2d(num_class)

#        
#        self.dirichlet_linear = nn.Linear(num_class, num_class*2)
#        self.binary_linear = nn.Linear(num_class*2, 2)

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))
 #       self.binary_linear.weight.data.fill_(1)
 #       self.binary_linear.bias.data.zero_()
        pass
    def forward(self, logits, gt):
    
#        logits = logits.permute(0,2,3,1).view(-1, num_class)
#        gt = gt.view(-1)

        logits = logits.permute(0,2,3,1)
#        gt = gt.view(-1)
        
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)

#        probs = logits
        ln_probs = torch.log(probs+1e-16)


#        out = logits
##        print(ln_probs.shape)
#        tf_positive = self.binary_linear(self.dirichlet_linear(ln_probs))


        out = self.dirichlet_linear(ln_probs)
        out = self.bn0(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_1(out.permute(0,2,3,1))
        out = self.bn1(out.permute(0,3,1,2))
        out = self.relu(out)
        
        out = self.linear_2(out.permute(0,2,3,1))
        out = self.bn2(out.permute(0,3,1,2))
        out = self.relu(out)       

#        out = self.linear_3(out.permute(0,2,3,1))
#        out = self.bn3(out.permute(0,3,1,2))
#        out = self.relu(out)  
#        
#        out = self.linear_4(out.permute(0,2,3,1))
#        out = self.bn4(out.permute(0,3,1,2))
#        out = self.relu(out)          
#        
        
        tf_positive = self.binary_linear(out.permute(0,2,3,1))
        
        _, pred = torch.max(probs, dim=-1)
        
        mask = pred == gt
        
        return  tf_positive.permute(0,3,1,2), mask.long()
###############################################################
        
        
class Dirichlet_Mask_Model(nn.Module):
    def __init__(self):
        super(Dirichlet_Mask_Model, self).__init__()
        self.dirichlet_linear = nn.Linear(num_class, num_class)

    def weights_init(self):
        self.dirichlet_linear.weight.data.copy_(torch.eye(self.dirichlet_linear.weight.shape[0]))
        self.dirichlet_linear.bias.data.copy_(torch.zeros(*self.dirichlet_linear.bias.shape))
#        pass
    def forward(self, logits):
    
#        logits = logits.permute(0,2,3,1).view(-1, num_class)
#        gt = gt.view(-1)

        logits = logits.permute(0,2,3,1)
#        gt = gt.view(-1)
        
        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(logits)
        ln_probs = torch.log(probs+1e-10)
#        print(ln_probs.shape)

        return self.dirichlet_linear(ln_probs).permute(0,3,1,2)    