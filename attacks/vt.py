import torch
import torch.nn.functional as F

from .helper import update_and_clip

__all__ = ["VT", "VTBaseline"]

class VT(object):
    def __init__(self, args, **kwargs):
        self.sample_times = args.sample_times
        self.beta = args.beta
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        v = 0
        for i in range(args.steps):
            adv_img.requires_grad_(True)
            loss = F.cross_entropy(model(adv_img), label, reduction="mean")
            adv_grad = torch.autograd.grad(loss, adv_img)[0].data
            input_grad = adv_grad + v
            
            # Calculate Gradient Variance
            neighbor_grad = 0
            for _ in range(self.sample_times):
                if args.constraint == "linf":
                    neighbor_img = adv_img.detach() + self.beta * adv_img.new(adv_img.size()).uniform_(-args.epsilon,args.epsilon)
                elif args.constraint == "l2":
                    neighbor_img = adv_img.detach() + self.beta * args.epsilon * F.normalize(adv_img.new(adv_img.size()).uniform_(-1,1), dim=(1,2,3))
                else:
                    raise RuntimeError("unsupport constraint")
                neighbor_img.requires_grad_(True)
                loss_neighbor = F.cross_entropy(model(neighbor_img), label, reduction="mean")
                neighbor_grad += torch.autograd.grad(loss_neighbor, neighbor_img)[0].data
            v = neighbor_grad / self.sample_times - adv_grad
            
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss.item()))
        return adv_img


class VTBaseline(object):
    def __init__(self, args, **kwargs):
        self.sample_times = args.sample_times
        self.beta = args.beta
        
    def __call__(self, args, ori_img, label, model, verbose=True):
        adv_img = ori_img.clone()
        v = 0
        for i in range(args.steps):
            neighbor_grad = 0
            for _ in range(self.sample_times):
                if args.constraint == "linf":
                    neighbor_img = adv_img.detach() + self.beta * adv_img.new(adv_img.size()).uniform_(-args.epsilon,args.epsilon)
                elif args.constraint == "l2":
                    neighbor_img = adv_img.detach() + self.beta * args.epsilon * F.normalize(adv_img.new(adv_img.size()).uniform_(-1,1), dim=(1,2,3))
                else:
                    raise RuntimeError("unsupport constraint")
                neighbor_img.requires_grad_(True)
                loss = F.cross_entropy(model(neighbor_img), label, reduction="mean")
                neighbor_grad += torch.autograd.grad(loss, neighbor_img)[0].data
            adv_img = update_and_clip(ori_img, adv_img, neighbor_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, Loss {:.4f}".format(i, loss.item()))
        return adv_img
