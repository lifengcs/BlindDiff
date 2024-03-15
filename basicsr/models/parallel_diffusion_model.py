import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, SRMDPreprocessing
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.matlab_functions import imresize_batch
from basicsr.parallel_diffusion import create_diffusion
from basicsr.parallel_diffusion.timestep_sampler import create_named_schedule_sampler
from .base_model import BaseModel
from torch.nn import functional as F
import time

@MODEL_REGISTRY.register()
class PDiffusionModel(BaseModel):
    """Blind SR model for blind single image super-resolution."""

    def __init__(self, opt):
        super(PDiffusionModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.net_k = build_network(opt['network_k'])
        self.net_k = self.model_to_device(self.net_k)
        #self.net_g = torch.compile(self.net_g,mode="max-autotune")
        #self.print_network(self.net_g)
        self.scale=opt["scale"]
        self.total_time=0
        
        # define diffusion
        self.num_sampling_steps = 250
        self.diffusion = create_diffusion(str(self.num_sampling_steps))
        print("diffusion created")
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)
        print("schedule_sampler created")
        
        self.last_epoch_loss = 1e8
        if opt["nonblindsr"]:
            self.flag=True
            self.pca_matrix = torch.load(
        opt["pca_matrix_path"], map_location=lambda storage, loc: storage
    )
        else:
            self.flag=False
            self.pca_matrix=None

        # load pretrained models
        load_path_g = self.opt['path'].get('pretrain_network_g', None)
        load_path_k = self.opt['path'].get('pretrain_network_k', None)
        if load_path_g is not None:
            param_key = self.opt['path'].get('param_key_g', 'params_ema')
            #ema! 
            self.load_network(self.net_g, load_path_g, self.opt['path'].get('strict_load_g', True), 'params_ema')
            
        if load_path_k is not None:
            param_key = self.opt['path'].get('param_key_k', 'params_ema')
            #ema! 
            self.load_network(self.net_k, load_path_k, self.opt['path'].get('strict_load_g', True), 'params_ema')
            
        if self.is_train:
            self.init_training_settings()
            self.prepro = SRMDPreprocessing(scale=opt["scale"], cuda=True, **opt["degradation"],pca_matrix=self.pca_matrix)
        else:
            self.cri_ref=None
            self.cri_detail=True
            self.cri_ker=None
            self.cri_structure=True


    def init_training_settings(self):
        self.net_g.train()
        self.net_k.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            self.net_k_ema = build_network(self.opt['network_k']).to(self.device)
            #self.net_g_ema = torch.compile(self.net_g_ema,mode="max-autotune")
            # load pretrained modelnum_in_ch
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
            self.net_k_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
            
        if train_opt.get('refine_opt'):
            self.cri_ref = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_ref = None

        if train_opt.get('ker_opt'):
            self.cri_ker = build_loss(train_opt['ker_opt']).to(self.device)
        else:
            self.cri_ker = None

        if train_opt.get('detail_opt'):
            self.cri_detail = build_loss(train_opt['detail_opt']).to(self.device)
        else:
            self.cri_detail = None

        if train_opt.get('structure_opt'):
            self.cri_structure = build_loss(train_opt['structure_opt']).to(self.device)
        else:
            self.cri_structure = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        #if self.cri_pix is None and self.cri_perceptual is None:
        #    raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        for k, v in self.net_k.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        #self.lq = data['lq'].to(self.device)
        if 'lq' in data and self.flag==False:
            print("validating")
            self.lq = data['lq'].to(self.device)
        else:
            self.lq, self.ker_code, self.ker, self.structure_gt, _ = self.prepro(self.gt, True)
            self.detail_gt=self.gt-self.structure_gt
            self.lq=(self.lq * 255).round() / 255
            self.ker = self.ker.unsqueeze(1)
            
            
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        t, weights = self.schedule_sampler.sample(self.lq.shape[0], self.device)
        model_kwargs = dict(cond=self.lq)
        loss = self.diffusion.training_losses(self.net_g,self.net_k,self.gt,self.ker,t,self.scale,model_kwargs=model_kwargs)
        l_total = (loss["loss"] * weights).mean()
        #print(l_total)
        loss_dict = OrderedDict()
        loss_dict['mse_g'] = loss["mse_g"]
        loss_dict['mse_k'] = loss["mse_k"]
        loss_dict['recon_g'] = loss["recon_g"]
        loss_dict['recon_k'] = loss["recon_k"]
        # pixel loss
        l_total.backward()
        if l_total<100*self.last_epoch_loss:
            self.optimizer_g.step()
            self.last_epoch_loss=l_total
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(l_total))

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # for swinDF
        window_size = 2#self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        model_kwargs = dict(cond=img)
        b, c, h, w = img.shape
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            z = torch.randn([b,c,h*scale,w*scale]).to(self.device)
            z_k = torch.randn([b,1,21,21]).to(self.device)
            self.output = self.diffusion.p_sample_loop(self.net_g_ema,self.net_k_ema,z.shape,z_k.shape,z,z_k,clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device)
        else:
            self.net_g.eval()
            z = torch.randn([b,c,h*scale,w*scale]).to(self.device)
            self.output = self.diffusion.p_sample_loop(self.net_g,z.shape,z,clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=self.device)
                
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if hasattr(self, 'structure'):
                structure_img = tensor2img([visuals['structure']])
            if hasattr(self, 'detail'):
                detail_img = tensor2img([visuals['detail']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                if hasattr(self, 'detail'):
                    save_detail_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_detail.png')
                    imwrite(detail_img, save_detail_path)
                if hasattr(self, 'structure'):
                    save_structure_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_structure.png')
                    imwrite(structure_img, save_structure_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'detail'):
            out_dict['detail'] = self.detail.detach().cpu()
        if hasattr(self, 'structure'):
            out_dict['structure'] = self.structure.detach().cpu() 
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if hasattr(self, 'net_k_ema'):
            self.save_network([self.net_k, self.net_k_ema], 'net_k', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_k, 'net_k', current_iter)
        self.save_training_state(epoch, current_iter)
