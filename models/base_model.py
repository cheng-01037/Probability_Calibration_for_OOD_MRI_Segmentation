"""
From the official cycleGAN implementation
"""
import os
import torch
import pdb



class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.snapshot_dir = opt.snapshot_dir
        self.pred_dir = opt.pred_dir

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def lr_update(self):
        for _scheduler in self.schedulers:
            _scheduler.step()

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.snapshot_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # load a specific network directly
    def load_network_by_fid(self, network, fid):
        network.load_state_dict(torch.load(fid))
        print(f'Load: network {fid} as been loaded')

    # copy paste things, copying from save_network and load_networl
    # helper saving function that can be used by subclasses
    def save_optimizer(self,optimizer, optimizer_label, epoch_label, gpu_ids):
        save_filename = '%s_optim_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.snapshot_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = '%s_optim_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.load_dir, save_filename)
        optimizer.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def as_np(self, data):
        return data.cpu().data.numpy()

    # added from new cycleGAN code
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
