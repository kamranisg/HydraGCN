import os
import torch

from torch.utils.tensorboard import SummaryWriter


class Callbacks:
    """Visualize results in tensorboard."""
    def __init__(self, args, fpath=None):

        if fpath is None:
            fpath = './outputs/{}/runs/{}_{}SimulatedMissing_{}SupervisionRate/'.format(args.dataset.name,
                                                                                        args.model_name,
                                                                                        args.p_missing,
                                                                                        args.supervision_rate)
            if not os.path.exists(fpath):
                print('Tensorboard Directory does not exist! Creating directory {}'.format(fpath))
                os.makedirs(fpath)
        self.args = args
        self.writer = SummaryWriter(fpath)
        self.max_image = 5
        self.epoch_counter = 0

        # Counter used to specify x-axis when plotting in tensorboard for on_batch_end
        self.on_batch_end_counter = {x: 0 for x in ['train', 'val', 'test']}

    def on_batch_end(self, phase, fold, output_list, target_list, loss_list):
        if self.epoch_counter % 30 == 0:
            loss_dict = {}
            with torch.no_grad():
                cur_batch_counter = self.on_batch_end_counter[phase]
                for i, output in enumerate(output_list):
                    # Log scalars in tensorboard
                    cur_loss = loss_list[i]
                    cur_loss_name = cur_loss['loss'].__class__.__name__

                    # Store loss name
                    if cur_loss_name not in loss_dict.keys():
                        loss_dict[cur_loss_name] = 1
                    else:
                        loss_dict[cur_loss_name] += 1
                        cur_loss_name += str(loss_dict[cur_loss_name] - 1)

                    log_str = '{}/fold{}/{}'.format(phase, fold, cur_loss_name)
                    cur_val = cur_loss['loss'](output, target_list[i]) * cur_loss['wts']
                    #self.writer.add_scalar(log_str, cur_val.item(), cur_batch_counter)
                    self.writer.add_scalar(log_str, cur_val, cur_batch_counter) ## changed by Alex according to branch: ...ekin
                    # Plot images in tensorboard
                    cur_target_keyname = cur_loss['loss'].target_keyname
                    cur_pred_keyname = cur_loss['loss'].pred_keyname
                    if cur_target_keyname is not None:

                        # In full batch mode we use indices to seperate train, val, and test
                        if output.get('indices', None) is not None:
                            indices_ = output['indices']
                            cur_target = output[cur_target_keyname][indices_]
                            cur_pred = output[cur_pred_keyname][indices_]
                        else:
                            raise NotImplementedError('indices are expected, please check if this is fully working'
                                                      'when using batch mode instead of full mode.')
                            cur_target = output[cur_target_keyname]
                            cur_pred = output[cur_pred_keyname]

                        cur_target = cur_target[:self.max_image]
                        cur_pred = cur_pred[:self.max_image]

                        # Reshape sequential data as images for now
                        if len(cur_pred.shape) == 3 and len(cur_pred.shape) == 3:
                            cur_target = cur_target.unsqueeze(1)
                            cur_pred = cur_pred.unsqueeze(1)

                        # Plot images
                        if len(cur_pred.shape) == 4:
                            cur_label = str(target_list[0][indices_][:self.max_image].cpu().numpy().tolist())
                            # print(cur_label)
                            self.writer.add_images(log_str + '/pred_{}_{}'.format(cur_label, cur_loss_name), cur_pred,
                                                   cur_batch_counter)

                            # Just plot target image once
                            if cur_batch_counter == 0:
                                self.writer.add_images(log_str + '/target_{}_{}'.format(cur_label, cur_loss_name),
                                                       cur_target,
                                                       cur_batch_counter)
                self.on_batch_end_counter[phase] += 1

    def on_epoch_end(self):
        """When a single epoch is done this callback will be called."""
        self.epoch_counter += 0
        self.writer.flush()

    def on_fold_end(self):
        """When a single fold from k-fold loop is done this is called."""
        # Reset counters which are mainly used in Tensorboard visualization.
        self.epoch_counter = 0
        self.on_batch_end_counter = {x: 0 for x in ['train', 'val', 'test']}
