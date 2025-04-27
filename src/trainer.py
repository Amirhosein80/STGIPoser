import os

import matplotlib.pyplot as plt
import comet_ml
import tensorboardX as tb
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn.functional import l1_loss

from src.base_model import rot_mat2r6d
from src.logger import get_logger, get_train_logger
from src.utils import (
    AverageMeter,
    angle_between,
    resume,
    save,
    set_seed,
    to_device,
)

logger = get_logger(__name__)


class Trainer:
    """
    A class to train models.

    Parameters:
        dataloaders: train and validation dataloaders
        model: model to train
        criterion: loss function
        optimizer: model optimizer
        scheduler: learning rate scheduler
        configs: model configurations
        experiment: comet_ml.Experiment

    Example:
    >>> dataloaders = {
            "train": DataLoader(train_dataset, batch_size=16, shuffle=True),
            "valid": DataLoader(valid_dataset, batch_size=16, shuffle=False),
        }
    >>> model = Model()
    >>> criterion = nn.MSELoss()
    >>> optimizer = optim.Adam(model.parameters(), lr=0.001)
    >>> scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    >>> configs = {
            "training": {
                "epochs": 100,
                "device": "cuda",
                "exp_name": "exp_name",
            }
        }
    >>> comet = set_comet(configs)
    >>> trainer = Trainer(dataloaders, model, criterion, optimizer, scheduler, configs, comet)
    >>> trainer.train()

    """

    def __init__(
        self,
        dataloaders: dict,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        configs: dict,
        experiment: comet_ml.Experiment,
    ) -> None:

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.experiment = experiment

        self.configs = configs

        self.num_epochs = configs["training"]["epochs"]
        self.dataloaders = dataloaders
        self.device = configs["training"]["device"]
        self.aux_weights = configs["loss"]["aux_weight"]
        self.log_dir = configs["log_dir"]
        self.name = configs["training"]["experimnet_name"]
        self.grad_norm = configs["optimizer"]["grad_norm"]
        self.model_name = configs["training"]["model_name"]
        self.use_translation = configs["model"]["use_translation"]
        self.use_imu_aux = configs["model"]["use_imu_aux"]
        self.sip_joints = configs["smpl"]["sip_joints"]


        self.metric_trans = AverageMeter()
        self.metric_pose = AverageMeter()

        self.loss_train = AverageMeter()
        self.loss_train_pose = AverageMeter()
        self.loss_train_trans = AverageMeter()
        self.loss_train_aux = AverageMeter()

        self.loss_valid = AverageMeter()
        self.loss_valid_pose = AverageMeter()
        self.loss_valid_trans = AverageMeter()
        self.loss_valid_aux = AverageMeter()

        self.train_hist = {
            "pose_loss": [],
            "trans_loss": [],
        }

        self.valid_hist = {
            "pose_loss": [],
            "trans_loss": [],
            "SIP_error": [],
            "Trans_error": [],
        }

        self.train_logger = get_train_logger(
            __name__, experiment_name=self.name, log_dir=self.log_dir
        )

        if configs["training"]["resume"]:
            (
                self.model,
                self.optimizer,
                self.scheduler,
                self.start_epoch,
                _,
                self.best_acc,
                self.train_hist,
                self.valid_hist,
            ) = resume(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                configs=configs,
            )
        else:
            self.start_epoch = 0
            self.best_acc = 120.0

    def one_step(self, datas: dict, training: bool = True) -> None:
        """
        One training or validation step.

        Parameters:
            datas: data dict
            training: is training
        """

        if training:

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(datas)
                main_loss, pose_loss, trans_loss, aux_loss = self.criterion(
                    outputs, datas
                )


            self.optimizer.zero_grad()
            main_loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

            self.optimizer.step()

            torch.cuda.synchronize()

            self.loss_train.update(main_loss)
            self.loss_train_pose.update(pose_loss)
            self.loss_train_trans.update(trans_loss)
            self.loss_train_aux.update(aux_loss)

        else:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                per_pose, per_trn = self.model.forward_offline(
                    imu_acc=datas["imu_acc"][0],
                    imu_ori=datas["imu_ori"][0],
                    uwb=datas["uwb"][0],
                )
            
                tar_grot = datas["grot"][0]
                per_grot = self.model.smpl_model.forward_kinematics_R(per_pose)
            
                self.metric_pose.update(angle_between(per_grot[:, self.sip_joints], tar_grot[:, self.sip_joints]))

                valid_output = {"out_rot": rot_mat2r6d(per_grot.unsqueeze(0)[:, :, self.model.reduced[1:]])}
                valid_target = {"grot": tar_grot.unsqueeze(0)}

                if self.use_translation:
                    tar_trn = datas["trans"].reshape(-1, 3)
                    per_trn = per_trn.reshape(-1, 3)
                    self.metric_trans.update(l1_loss(per_trn, tar_trn))
                    valid_output["out_trans"] = per_trn
                    valid_target["trans"] = tar_trn

                main_loss, pose_loss, trans_loss, aux_loss = self.criterion(
                    valid_output, valid_target
                )

            self.loss_valid.update(main_loss)
            self.loss_valid_pose.update(pose_loss)
            self.loss_valid_trans.update(trans_loss)
            self.loss_valid_aux.update(aux_loss)

    def train_one_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        """
        One training epoch

        Parameters:
            epoch: number of epoch

        Returns:
            train loss: the average loss of the training set
            train pose loss: the average pose loss of the training set
            train trans loss: the average translation loss of the training set
            train aux loss: the average auxiliary loss of the training set
        """
        self.model.train()

        self.loss_train.reset()
        self.loss_train_pose.reset()
        self.loss_train_trans.reset()
        self.loss_train_aux.reset()

        with tqdm.tqdm(self.dataloaders["train"], unit="batch") as tepoch:
            for datas in tepoch:
                datas = to_device(data=datas, device=self.device)
                self.one_step(datas, training=True)
                tepoch.set_postfix(
                    epoch=epoch,
                    main_loss=self.loss_train.avg,
                    pose_loss=self.loss_train_pose.avg,
                    trans_loss=self.loss_train_trans.avg,
                    aux_loss=self.loss_train_aux.avg,
                    phase="Training",
                )
            if self.scheduler is not None:
                self.scheduler.step()

        return (
            self.loss_train.avg,
            self.loss_train_pose.avg,
            self.loss_train_trans.avg,
            self.loss_train_aux.avg,
        )

    def valid_one_epoch(self) -> tuple[float, float, float, float, float, float]:
        """
        One validation epoch

        Returns:
            valid loss: the average loss of the validation set
            valid pose loss: the average pose loss of the validation set
            valid trans loss: the average translation loss of the validation set
            valid aux loss: the average auxiliary loss of the validation set
            valid pose metric: the average pose metric of the validation set
            valid trans metric: the average translation metric of the validation set
        """
        self.model.eval()

        self.loss_valid.reset()
        self.loss_valid_pose.reset()
        self.loss_valid_trans.reset()
        self.loss_valid_aux.reset()

        self.metric_pose.reset()
        self.metric_trans.reset()

        with tqdm.tqdm(self.dataloaders["valid"], unit="batch") as tepoch:
            with torch.inference_mode():
                for datas in tepoch:
                    datas = to_device(data=datas, device=self.device)
                    self.one_step(datas, training=False)
                    tepoch.set_postfix(
                        main_loss=self.loss_valid.avg,
                        pose_loss=self.loss_valid_pose.avg,
                        trans_loss=self.loss_valid_trans.avg,
                        aux_loss=self.loss_valid_aux.avg,
                        sip_error=self.metric_pose.avg, 
                        phase="Evaluation",
                    )

        return (
            self.loss_valid.avg,
            self.loss_valid_pose.avg,
            self.loss_valid_trans.avg,
            self.loss_valid_aux.avg,
            self.metric_pose.avg,
            self.metric_trans.avg,
        )

    def train(self) -> None:
        """
        Train the model.
        """
        logger.info(f"Strat experiment Name: {self.name} with model {self.model_name}")
        self.train_logger.info(
            f"Strat training model {self.model_name} with {self.num_epochs} epochs from {self.start_epoch}"
        )

        for epoch in range(self.start_epoch, self.num_epochs):
            self.experiment.log_current_epoch(epoch)
            set_seed(epoch)
            with self.experiment.train():
                train_loss, train_pose_loss, train_trans_loss, train_aux_loss = (
                    self.train_one_epoch(epoch)
                )
            with self.experiment.test():
                (
                    valid_loss,
                    valid_pose_loss,
                    valid_trans_loss,
                    valid_aux_loss,
                    valid_pose_metric,
                    valid_trans_metric,
                ) = self.valid_one_epoch()

            self.train_hist["pose_loss"].append(train_pose_loss)
            self.train_hist["trans_loss"].append(train_trans_loss)
            self.valid_hist["pose_loss"].append(valid_pose_loss)
            self.valid_hist["trans_loss"].append(valid_trans_loss)

            self.valid_hist["SIP_error"].append(valid_pose_metric)
            self.valid_hist["Trans_error"].append(valid_trans_metric)

            if valid_pose_metric < self.best_acc:
                self.best_acc = valid_pose_metric
                save(
                    model=self.model,
                    acc=valid_pose_metric,
                    best_acc=self.best_acc,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    train_hist=self.train_hist,
                    valid_hist=self.valid_hist,
                    name=f"best_{self.name}",
                    log_dir=self.log_dir,
                )
                self.experiment.log_model(
                    self.model_name,
                    os.path.join(self.log_dir, f"checkpoint/best_{self.name}.pth"),
                )

                self.train_logger.info(f"Model saved! at epoch {epoch} in {self.log_dir}")
                
            self.train_logger.info(
                f"Train >>> Epoch {epoch} Training Loss: {train_loss:.4}, Train Pose Loss: {train_pose_loss:.4}, Train Trans Loss: {train_trans_loss:.4}, Train Aux Loss: {train_aux_loss:.4}"
            )


            self.train_logger.info(
                f"Valid >>> Epoch {epoch} Valid Loss: {valid_loss:.4}, Valid Pose Loss: {valid_pose_loss:.4}, Valid Trans Loss: {valid_trans_loss:.4}, Valid Aux Loss: {valid_aux_loss:.4}, Valid SIP Error: {valid_pose_metric:.4}, Valid Trans Error: {valid_trans_metric:.4}, Best SIP Error: {self.best_acc:.4}"
            )
            
            save(
                model=self.model,
                best_acc=self.best_acc,
                acc=valid_pose_metric,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                train_hist=self.train_hist,
                valid_hist=self.valid_hist,
                name=f"last_{self.name}",
                log_dir=self.log_dir,
            )

            self.experiment.log_metrics(
                {
                    "pose_loss_train": self.train_hist["pose_loss"][-1],
                    "trans_loss_train": self.train_hist["trans_loss"][-1],
                    "pose_loss_valid": self.valid_hist["pose_loss"][-1],
                    "trans_loss_valid": self.valid_hist["trans_loss"][-1],
                    "SIP_error": self.valid_hist["SIP_error"][-1],
                    "Trans_error": self.valid_hist["Trans_error"][-1],
                },
                epoch=epoch,
            )

            plt.clf()
            plt.plot(self.train_hist["pose_loss"], label="Train Pose Loss")
            plt.plot(self.valid_hist["pose_loss"], label="Valid Pose Loss")
            plt.title("Pose Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"pose_loss_log.png"))

            plt.clf()
            plt.plot(self.train_hist["trans_loss"], label="Train Trans Loss")
            plt.plot(self.valid_hist["trans_loss"], label="Valid Trans Loss")
            plt.title("Trans Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"trans_loss_log.png"))

            plt.clf()
            plt.plot(self.valid_hist["SIP_error"], label="SIP Error")
            plt.title("SIP Error")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"SIP_error_log.png"))

            plt.clf()
            plt.plot(self.valid_hist["Trans_error"], label="Trans Error")
            plt.title("Trans Error")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f"Trans_error_log.png"))

            print()
