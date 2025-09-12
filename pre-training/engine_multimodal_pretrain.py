import math
import sys
import random
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
from timm.data import Mixup
import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.options import logger

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    accum_iter, batch_size = args.accum_iter, args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    omics_feats = None
    image_feats = None
    omics_embeds = []
    image_embeds = []
    omics_mask, idx_mask = [], []
    k = -1
    # for data_iter_step, (vec_patches, X_mrna, X_mirna, X_meth) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (regions, X_mrna, X_mirna, X_meth) in enumerate(data_loader):

        # if data_iter_step > 0:
        #     metric_logger.update(lr=0.0001)
        #     break

        # we use a per iteration (instead of per epoch) lr scheduler
        if (data_iter_step+1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        regions = regions.to(device, non_blocking=True)
        X_mrna = X_mrna.to(device, non_blocking=True)
        X_mirna = X_mirna.to(device, non_blocking=True)
        X_meth = X_meth.to(device, non_blocking=True)

        # random mask omics data
        ri = random.randint(1, 3)
        if ri == 1:
            omics_mask.append(X_mrna[0])
            idx_mask.append(1)
            X_mrna = torch.ones([1, X_mrna.shape[1]], dtype=torch.float32)\
                .to(device, non_blocking=True)
        elif ri == 2:
            omics_mask.append(X_mirna[0])
            idx_mask.append(2)
            X_mirna = torch.ones([1, X_mirna.shape[1]], dtype=torch.float32)\
                .to(device, non_blocking=True)
        elif ri == 3:
            omics_mask.append(X_meth[0])
            idx_mask.append(3)
            X_meth = torch.ones([1, X_meth.shape[1]], dtype=torch.float32)\
                .to(device, non_blocking=True)
        else:
            pass

        #
        with torch.cuda.amp.autocast():
            samples = [regions, X_mrna, X_mirna, X_meth]
            img_cls, omics_cls, image_embed, omics_embed = model(samples)

        k += 1
        if k == 0:
            omics_feats = omics_cls
            image_feats = img_cls
        else:
            omics_feats = torch.cat((omics_feats, omics_cls), 0)
            image_feats = torch.cat((image_feats, img_cls), 0)
        image_embeds.append(image_embed)
        omics_embeds.append(omics_embed)

        if k == accum_iter - 1:
            k = -1

            # calculating contrastive loss by referring CLIP code
            # normalized features
            omics_features = omics_feats / omics_feats.norm(dim=1, keepdim=True)
            image_features = image_feats / image_feats.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            sim_i2o = logit_scale * image_features @ omics_features.t()
            sim_o2i = sim_i2o.t()
            contrastive_labels = torch.arange(batch_size, device=device)
            loss_poc = (F.cross_entropy(sim_i2o, contrastive_labels) + \
                   F.cross_entropy(sim_o2i, contrastive_labels))/2

            with torch.no_grad():

                weights_i2o = F.softmax(sim_i2o[:, :batch_size], dim=1)
                weights_o2i = F.softmax(sim_o2i[:, :batch_size], dim=1)

                weights_i2o.fill_diagonal_(0)
                weights_o2i.fill_diagonal_(0)

            # calculate the pathology-omics match loss
            # for positive samples
            logits_mask, logits_cls = [], []
            for b in range(batch_size):
                logit_mask, logit_cls = model.path_guided_omics_encoder(image_embeds[b], omics_embeds[b], idx_mask[b])
                logits_mask.append(logit_mask[0])
                logits_cls.append(logit_cls[0])

            # calculate the loss of masked omics modeling first
            omics_mask = torch.stack(omics_mask, dim=0)
            logits_mask = torch.stack(logits_mask, dim=0)
            loss_mom = (logits_mask - omics_mask) ** 2
            loss_mom = loss_mom.mean()

            try:
                # select a negative image for each omics
                for b in range(batch_size):
                    neg_idx = torch.multinomial(weights_o2i[b], 1).item()
                    _, logit_cls = model.path_guided_omics_encoder(image_embeds[neg_idx], omics_embeds[b], idx_mask[b])
                    logits_cls.append(logit_cls[0])

                # select a negative omics for each image
                for b in range(batch_size):
                    neg_idx = torch.multinomial(weights_i2o[b], 1).item()
                    _, logit_cls = model.path_guided_omics_encoder(image_embeds[b], omics_embeds[neg_idx], idx_mask[neg_idx])
                    logits_cls.append(logit_cls[0])

            except Exception as e:
                omics_embeds = []
                image_embeds = []
                omics_mask, idx_mask = [], []
                logger.info(f"Exception: {e.args}")
                continue

            itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                                   dim=0).to(device, non_blocking=True)

            logits_cls = torch.stack(logits_cls, dim=0)
            loss_pom = F.cross_entropy(logits_cls, itm_labels)

            omics_embeds = []
            image_embeds = []
            omics_mask, idx_mask = [], []

            ### total loss
            loss = loss_poc * 1.0 + loss_pom * 6.0 + loss_mom * 3.0

            ### 6 combinations
            # loss = loss_poc * 1.0 + loss_pom * 0.0 + loss_mom * 0.0
            # loss = loss_poc * 0.0 + loss_pom * 6.0 + loss_mom * 0.0
            # loss = loss_poc * 0.0 + loss_pom * 0.0 + loss_mom * 3.0

            # loss = loss_poc * 0.0 + loss_pom * 6.0 + loss_mom * 3.0
            # loss = loss_poc * 1.0 + loss_pom * 0.0 + loss_mom * 3.0
            # loss = loss_poc * 1.0 + loss_pom * 6.0 + loss_mom * 0.0

            metric_logger.update(loss_poc=loss_poc.item() * 1.0)
            metric_logger.update(loss_pom=loss_pom.item() * 6.0)
            metric_logger.update(loss_mom=loss_mom.item() * 3.0)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)

            logger.info(f"index:{data_iter_step+1}, loss: {loss}")

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, model

