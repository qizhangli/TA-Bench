import torch
import torch.nn.functional as F

from .helper import update_and_clip

__all__ = ["TAIG", "TAIGBaseline"]

def compute_ig(inputs,label,model,epsilon,constraint,steps=20):
    baseline = torch.zeros(inputs.shape).to(inputs.device)
    avg_grads = 0
    for i in range(steps+1):
        scaled_inputs = (float(i) / steps) * inputs
        if constraint == "linf":
            scaled_inputs = scaled_inputs + scaled_inputs.new(scaled_inputs.size()).uniform_(-epsilon,epsilon)
        elif constraint == "l2":
            scaled_inputs = scaled_inputs + epsilon * F.normalize(scaled_inputs.new(scaled_inputs.size()).uniform_(-1,1), dim=(1,2,3))
        else:
            raise RuntimeError("unsupport constraint")
        scaled_inputs.requires_grad_(True)
        att_out = model(scaled_inputs)
        score = att_out[torch.arange(len(scaled_inputs)), label]
        loss = -torch.mean(score)
        grads = torch.autograd.grad(loss, scaled_inputs)[0].data
        avg_grads += grads
    avg_grads /= steps
    integrated_grad = scaled_inputs * avg_grads
    IG = integrated_grad.detach()
    return IG, loss

class TAIG(object):
    def __init__(self, args, **kwargs):
        self.sample_times = args.sample_times
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad, loss = compute_ig(adv_img, label, model, epsilon=args.epsilon, constraint=args.constraint, steps=self.sample_times)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss.item()))
        return adv_img


def compute_aug_grad(inputs,label,model,epsilon,constraint,steps=20):
    baseline = torch.zeros(inputs.shape).to(inputs.device)
    avg_grads = 0
    for i in range(steps+1):
        scaled_inputs = (float(i) / steps) * inputs
        if constraint == "linf":
            scaled_inputs = scaled_inputs + scaled_inputs.new(scaled_inputs.size()).uniform_(-epsilon,epsilon)
        elif constraint == "l2":
            scaled_inputs = scaled_inputs + epsilon * F.normalize(scaled_inputs.new(scaled_inputs.size()).uniform_(-1,1), dim=(1,2,3))
        else:
            raise RuntimeError("unsupport constraint")
        scaled_inputs.requires_grad_(True)
        att_out = model(scaled_inputs)
        loss = F.cross_entropy(att_out, label)
        grads = torch.autograd.grad(loss, scaled_inputs)[0].data
        avg_grads += grads
    avg_grads /= steps
    return avg_grads, loss

class TAIGBaseline(TAIG):
    def __init__(self, args, **kwargs):
        self.sample_times = args.sample_times
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad, loss = compute_aug_grad(adv_img, label, model, epsilon=args.epsilon, constraint=args.constraint, steps=self.sample_times)
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss.item()))
        return adv_img
