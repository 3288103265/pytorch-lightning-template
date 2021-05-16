import os
from numpy.lib.type_check import real
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_generator_128 import ResnetGenerator128
from model.combine_discriminator_128 import CombineDiscriminator128
from model.loss import LossManager, VGGLoss, Conditional_Contrastive_loss, make_mask, set_temperature
from model.utils import imsave, calculate_fid_given_paths


class LostGan(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.img_size = 128
        self.z_dim = 128

        self.num_classes = 184 if self.hparams.dataset == 'coco' else 179
        self.num_obj = 8 if self.hparams.dataset == 'coco' else 31
        self.netG = ResnetGenerator128(
            num_classes=self.num_classes, output_dim=3, use_trans_enc=self.hparams.use_trans_enc)
        self.netD = CombineDiscriminator128(num_classes=self.num_classes)
        self.configure_loss()

    def configure_loss(self):
        self.lamb_obj = 1.0
        self.lamb_img = 0.1
        self.lamb_contra = 1.0
        self.lamb_app = 1.0
        self.vgg_loss = VGGLoss()
        self.l1_loss = nn.L1Loss()
        self.contra_loss = Conditional_Contrastive_loss(
            self.hparams.batch_size, pos_collected_numerator=True)

    def forward(self, z, bbox, label, src_mask):
        return self.netG(z, bbox, y=label.squeeze(
            dim=-1), src_mask=src_mask)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, label, bbox = batch
        src_mask = torch.bmm(label.unsqueeze(2), label.unsqueeze(1))
        src_mask = src_mask != 0

        z = torch.randn(real_images.size(
            0), self.num_obj, self.z_dim).to(real_images)
        fake_images = self(z, bbox, y=label.squeeze(
            dim=-1), src_mask=src_mask)

        if optimizer_idx == 0:
            self.d_loss = LossManager()
            d_out_real, d_out_robj, d_out_robj_app, obj_feat_real, cls_feat_proxy_real, label_proxy_real = self.netD(
                real_images, bbox, label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
            d_loss_robj_app = torch.nn.ReLU()(1.0 - d_out_robj_app).mean()

            d_out_fake, d_out_fobj, d_out_fobj_app, obj_feat_fake, cls_feat_proxy_fake, label_proxy_fake = self.netD(
                fake_images.detach(), bbox, label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
            d_loss_fobj_app = torch.nn.ReLU()(1.0 + d_out_fobj_app).mean()

            real_cls_mask = make_mask(
                label_proxy_real, self.num_classes, mask_negatives=True)

            real_cls_mask = real_cls_mask.T
            contra_loss_real = self.contra_loss(
                obj_feat_real,
                cls_feat_proxy_real,
                real_cls_mask,
                label_proxy_real,
                1.0,
                margin=0.0
            ).mean()

            d_loss_obj = d_loss_fobj + d_loss_robj
            self.d_loss.add_loss(d_loss_obj, 'd_loss/obj', self.lamb_obj)
            d_loss_img = d_loss_real + d_loss_fake
            self.d_loss.add_loss(d_loss_img, 'd_loss/img', self.lamb_img)
            d_loss_app = d_loss_robj_app + d_loss_fobj_app
            self.d_loss.add_loss(d_loss_app, 'd_loss/app', self.lamb_app)
            self.d_loss.add(contra_loss_real,
                            'd_loss/contra_real', self.lamb_contra)

            self.log_dict(self.d_loss.all_losses)
            return self.d_loss.total_loss

        # update G
        if ((batch_idx+1) % self.hparams.giter) == 0 and optimizer_idx == 1:

            g_out_fake, g_out_obj, g_out_obj_app, obj_feat_fake, cls_feat_proxy_fake, label_proxy_fake = self.netD(
                fake_images, bbox, label)
            g_loss_fake = - g_out_fake.mean()
            g_loss_obj = - g_out_obj.mean()
            g_loss_obj_app = - g_out_obj_app.mean()

            pixel_loss = self.l1_loss(fake_images, real_images).mean()
            feat_loss = self.vgg_loss(fake_images, real_images).mean()
            # contrastive loss
            fake_cls_mask = make_mask(
                label_proxy_fake, self.num_classes, mask_negatives=True)
            fake_cls_mask = fake_cls_mask.T
            contra_loss_fake = self.contra_loss(obj_feat_fake,
                                                cls_feat_proxy_fake,
                                                fake_cls_mask,
                                                label_proxy_fake,
                                                1.0,
                                                margin=0.0
                                                ).mean()

            self.g_loss.add_loss(g_loss_obj, 'g_loss/obj', self.lamb_obj)
            self.g_loss.add_loss(g_loss_fake, 'g_loss/img', self.lamb_img)
            self.g_loss.add_loss(g_loss_obj_app, 'g_loss/app', self.lamb_app)
            self.g_loss.add_loss(pixel_loss, 'g_loss/pixel', 1.0)
            self.g_loss.add_loss(feat_loss, 'g_loss/feat', 1.0)
            self.g_loss.add(contra_loss_fake,
                            'g_loss/contra_fake', self.lamb_contra)
            self.log_dict(self.g_loss.all_losses)
            return self.g_loss.total_loss

    def validation_step(self, batch, batch_idx):
        real_images, label, bbox = batch
        log_dir = self.logger.log_dir

        # save GT
        gt_dir = os.path.join(log_dir, "samples_" +
                              str(self.current_epoch), "images_gt")
        fake_dir = os.path.join(log_dir, "samples_" +
                                str(self.current_epoch), "images")

        if self.hparams.use_trans_enc:
            src_mask = torch.bmm(label.unsqueeze(2), label.unsqueeze(1))
            src_mask = src_mask != 0
            src_mask = src_mask.cuda()
        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        bbox = bbox.float().cuda()

        for s_i in range(self.hparams.sample_num):  # sample_num=5
            z_obj = torch.randn(self.hparams.batch_size, self.hparams.num_o, 128,
                                device=real_images.device)

            z_im = torch.randn(self.hparams.batch_size, 128,
                               device=real_images.device)
            fake_images = self(
                z_obj, bbox, z_im, label.squeeze(dim=-1), src_mask=src_mask)

            for j, img in enumerate(fake_images):
                imsave("{save_path}/sample{id}_{s_i}.jpg".format(save_path=fake_dir,
                                                                 id=batch_idx*self.hparams.batch_size+j, s_i=s_i), img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)

        for k, img in enumerate(real_images):
            imsave("{save_path}/sample{id}.jpg".format(save_path=gt_dir,
                                                       s_i=s_i, id=batch_idx*self.hparams.batch_size+k), img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)

    def validation_epoch_end(self, outputs):
        log_dir = self.logger.log_dir
        gt_dir = os.path.join(log_dir, "samples_" +
                              str(self.current_epoch), "images_gt")
        fake_dir = os.path.join(log_dir, "samples_" +
                                str(self.current_epoch), "images")
        
        paths = [fake_dir, gt_dir]
        # print('>>>calculating fid score...')
        # TODO: calculate fid using dp??
        return calculate_fid_given_paths(paths, batch_size=50, device=torch.cuda.current_device(), dims=2048)
        
        
        return super().validation_epoch_end(outputs)

    def test_step(self, *self.hparams, **kwself.hparams):
        return super().test_step(*self.hparams, **kwself.hparams)

    def configure_optimizers(self):
        d_lr = self.hparams.d_lr
        g_lr = self.hparams.g_lr
        gen_parameters = []
        for key, value in dict(self.netG.named_parameters()).items():
            if value.requires_grad:
                if 'mapping' in key:
                    gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
                else:
                    gen_parameters += [{'params': [value], 'lr': g_lr}]

        g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

        dis_parameters = []
        for key, value in dict(self.netD.named_parameters()).items():
            if value.requires_grad:
                dis_parameters += [{'params': [value], 'lr': d_lr}]
        d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))
        return [d_optimizer, g_optimizer], []
