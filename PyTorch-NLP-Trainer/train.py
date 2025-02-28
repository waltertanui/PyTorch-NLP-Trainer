# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : 390737991@qq.com
    @Date   : 2022-09-26 14:50:34
    @Brief  :
"""
import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import tensorboardX as tensorboard
from tqdm import tqdm
from torch.utils import data as data_utils
from core.dataloader import build_dataset
from core.models import build_models
from core.criterion.build_criterion import get_criterion
from core.utils import torch_tools, metrics, log
from pybaseutils import file_utils, config_utils
from pybaseutils.metrics import class_report


class Trainer(object):
    def __init__(self, cfg):
        torch_tools.set_env_random_seed()
        # 设置输出路径
        time = file_utils.get_time()
        flag = [n for n in [cfg.net_type, cfg.loss_type, cfg.flag, time] if n]
        cfg.work_dir = os.path.join(cfg.work_dir, "_".join(flag))
        cfg.model_root = os.path.join(cfg.work_dir, "model")
        cfg.log_root = os.path.join(cfg.work_dir, "log")
        file_utils.create_dir(cfg.work_dir)
        file_utils.create_dir(cfg.model_root)
        file_utils.create_dir(cfg.log_root)
        file_utils.copy_file_to_dir(cfg.config_file, cfg.work_dir)
        config_utils.save_config(cfg, os.path.join(cfg.work_dir, "setup_config.yaml"))
        self.cfg = cfg
        self.topk = self.cfg.topk
        # 配置GPU/CPU运行设备
        self.gpu_id = cfg.gpu_id
        self.device = torch.device("cuda:{}".format(cfg.gpu_id[0]) if torch.cuda.is_available() else "cpu")
        # 设置Log打印信息
        self.logger = log.set_logger(level="debug", logfile=os.path.join(cfg.log_root, "train.log"))
        # 构建训练数据和测试数据
        self.train_loader = self.build_train_loader()
        self.test_loader = self.build_test_loader()
        # 构建模型
        self.model = self.build_model()
        # 构建损失函数
        self.criterion = self.build_criterion()
        # 构建优化器
        self.optimizer = self.build_optimizer()
        # 构建学习率调整策略
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, cfg.milestones)
        # 使用tensorboard记录和可视化Loss
        self.writer = tensorboard.SummaryWriter(cfg.log_root)
        # 打印信息
        self.num_samples = len(self.train_loader.sampler)
        self.logger.info("=" * 60)
        self.logger.info("work_dir          :{}".format(cfg.work_dir))
        self.logger.info("config_file       :{}".format(cfg.config_file))
        self.logger.info("gpu_id            :{}".format(cfg.gpu_id))
        self.logger.info("main device       :{}".format(self.device))
        self.logger.info("num_samples(train):{}".format(self.num_samples))
        self.logger.info("num_classes       :{}".format(cfg.num_classes))
        self.logger.info("mean_num          :{}".format(self.num_samples / cfg.num_classes))
        self.logger.info("=" * 60)

    def build_optimizer(self, ):
        """build_optimizer"""
        if self.cfg.optim_type.lower() == "SGD".lower():
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.cfg.lr,
                                        momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optim_type.lower() == "Adam".lower():
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            optimizer = None
        return optimizer

    def build_train_loader(self, ) -> data_utils.DataLoader:
        """build_train_loader"""
        self.logger.info("build_train_loader,context_size:{}".format(self.cfg.context_size))
        dataset = build_dataset.load_dataset(data_type=self.cfg.data_type,
                                             filename=self.cfg.train_data,
                                             vocab_file=self.cfg.vocab_file,
                                             context_size=self.cfg.context_size,
                                             class_name=self.cfg.class_name,
                                             resample=self.cfg.resample,
                                             phase="train",
                                             shuffle=True)
        shuffle = True
        sampler = None
        self.logger.info("use resample:{}".format(self.cfg.resample))
        # if self.cfg.resample:
        #     weights = torch.DoubleTensor(dataset.classes_weights)
        #     sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        #     shuffle = False
        loader = data_utils.DataLoader(dataset=dataset, batch_size=self.cfg.batch_size, sampler=sampler,
                                       shuffle=shuffle, num_workers=self.cfg.num_workers)
        self.cfg.num_classes = dataset.num_classes
        self.cfg.num_embeddings = dataset.num_embeddings
        self.cfg.class_name = dataset.class_name
        file_utils.copy_file_to_dir(self.cfg.vocab_file, cfg.work_dir)
        return loader

    def build_test_loader(self, ) -> data_utils.DataLoader:
        """build_test_loader"""
        self.logger.info("build_test_loader,context_size:{}".format(cfg.context_size))
        dataset = build_dataset.load_dataset(data_type=self.cfg.data_type,
                                             filename=self.cfg.test_data,
                                             vocab_file=self.cfg.vocab_file,
                                             context_size=self.cfg.context_size,
                                             class_name=self.cfg.class_name,
                                             phase="test",
                                             resample=False,
                                             shuffle=False)
        loader = data_utils.DataLoader(dataset=dataset, batch_size=self.cfg.batch_size,
                                       shuffle=False, num_workers=self.cfg.num_workers)
        self.cfg.num_classes = dataset.num_classes
        self.cfg.num_embeddings = dataset.num_embeddings
        self.cfg.class_name = dataset.class_name
        return loader

    def build_model(self, ) -> nn.Module:
        """build_model"""
        self.logger.info("build_model,net_type:{}".format(self.cfg.net_type))
        model = build_models.get_models(net_type=self.cfg.net_type,
                                        num_classes=self.cfg.num_classes,
                                        num_embeddings=self.cfg.num_embeddings,
                                        context_size = self.cfg.context_size,
                                        embedding_dim=128,
                                        is_train=True,
                                        )
        if self.cfg.finetune:
            self.logger.info("finetune:{}".format(self.cfg.finetune))
            state_dict = torch_tools.load_state_dict(self.cfg.finetune)
            model.load_state_dict(state_dict)
        model = model.to(self.device)
        model = nn.DataParallel(model, device_ids=self.gpu_id, output_device=self.device)
        return model

    def build_criterion(self, ):
        """build_criterion"""
        self.logger.info(
            "build_criterion,loss_type:{}, num_embeddings:{}".format(self.cfg.loss_type, self.cfg.num_embeddings))
        criterion = get_criterion(self.cfg.loss_type, self.cfg.num_embeddings, device=self.device)
        # criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def train(self, epoch):
        """训练"""
        train_losses = metrics.AverageMeter()
        train_accuracy = {k: metrics.AverageMeter() for k in self.topk}
        self.model.train()  # set to training mode
        log_step = max(len(self.train_loader) // cfg.log_freq, 1)
        for step, data in enumerate(tqdm(self.train_loader)):
            inputs, target = data
            inputs, target = inputs.to(self.device), target.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, target)
            self.optimizer.zero_grad()  # 反馈
            loss.backward()
            self.optimizer.step()  # 更新
            train_losses.update(loss.cpu().data.item())
            # 计算准确率
            target = target.cpu()
            outputs = outputs.cpu()
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            pred_score, pred_index = torch.max(outputs, dim=1)
            acc = metrics.accuracy(outputs.data, target, topk=self.topk)
            for i in range(len(self.topk)):
                train_accuracy[self.topk[i]].update(acc[i].data.item(), target.size(0))
            if step % log_step == 0:
                lr = self.scheduler.get_last_lr()[0]  # 获得当前学习率
                topk_acc = {"top{}".format(k): v.avg for k, v in train_accuracy.items()}
                self.logger.info(
                    "train {}/epoch:{:0=3d},lr:{:3.4f},loss:{:3.4f},acc:{}".format(step, epoch, lr, train_losses.avg,
                                                                                    topk_acc))

        topk_acc = {"top{}".format(k): v.avg for k, v in train_accuracy.items()}
        self.writer.add_scalar("train-loss", train_losses.avg, epoch)
        self.writer.add_scalars("train-accuracy", topk_acc, epoch)
        self.logger.info("train epoch:{:0=3d},loss:{:3.4f},acc:{}".format(epoch, train_losses.avg, topk_acc))
        return topk_acc["top{}".format(self.topk[0])]

    def test(self, epoch):
        """测试"""
        test_losses = metrics.AverageMeter()
        test_accuracy = {k: metrics.AverageMeter() for k in self.topk}
        true_labels = np.ones(0)
        pred_labels = np.ones(0)
        self.model.eval()  # set to evaluates mode
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.test_loader)):
                inputs, target = data
                inputs, target = inputs.to(self.device), target.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)
                test_losses.update(loss.cpu().data.item())
                # 计算准确率
                target = target.cpu()
                outputs = outputs.cpu()
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                pred_score, pred_index = torch.max(outputs, dim=1)
                acc = metrics.accuracy(outputs.data, target, topk=self.topk)
                true_labels = np.hstack([true_labels, target.numpy()])
                pred_labels = np.hstack([pred_labels, pred_index.numpy()])

                for i in range(len(self.topk)):
                    test_accuracy[self.topk[i]].update(acc[i].data.item(), target.size(0))

        report = class_report.get_classification_report(true_labels, pred_labels, target_names=self.cfg.class_name)
        topk_acc = {"top{}".format(k): v.avg for k, v in test_accuracy.items()}
        lr = self.scheduler.get_last_lr()[0]  # 获得当前学习率
        self.writer.add_scalar("test-loss", test_losses.avg, epoch)
        self.writer.add_scalars("test-accuracy", topk_acc, epoch)
        self.logger.info("test  epoch:{:0=3d},lr:{:3.4f},loss:{:3.4f},acc:{}".format(epoch, lr, test_losses.avg, topk_acc))
        # self.logger.info("{}".format(report))
        return topk_acc["top{}".format(self.topk[0])]

    def run(self):
        """开始运行"""
        self.max_acc = 0.0
        for epoch in range(self.cfg.num_epochs):
            train_acc = self.train(epoch)  # 训练模型
            test_acc = self.test(epoch)  # 测试模型
            self.scheduler.step()  # 更新学习率
            lr = self.scheduler.get_last_lr()[0]  # 获得当前学习率
            self.writer.add_scalar("lr", lr, epoch)
            self.save_model(self.cfg.model_root, test_acc, epoch)
            self.logger.info("epoch:{}, lr:{}, train acc:{:3.4f}, test acc:{:3.4f}".
                             format(epoch, lr, train_acc, test_acc))

    def save_model(self, model_root, value, epoch):
        """保存模型"""
        # 保存最优的模型
        if value >= self.max_acc:
            self.max_acc = value
            model_file = os.path.join(model_root, "best_model_{:0=3d}_{:.4f}.pth".format(epoch, value))
            file_utils.remove_prefix_files(model_root, "best_model_*")
            torch.save(self.model.module.state_dict(), model_file)
            self.logger.info("save best   model file:{}".format(model_file))
        # 保存最新的模型
        name = "model_{:0=3d}_{:.4f}.pth".format(epoch, value)
        model_file = os.path.join(model_root, "latest_{}".format(name))
        file_utils.remove_prefix_files(model_root, "latest_*")
        torch.save(self.model.module.state_dict(), model_file)
        self.logger.info("save latest model file:{}".format(model_file))
        self.logger.info("-------------------------" * 4)


def get_parser():
    cfg_file = "configs/config.yaml"
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument("-c", "--config_file", help="configs file", default=cfg_file, type=str)
    parser.add_argument('--polyaxon', action='store_true', help='polyaxon', default=False)
    cfg = config_utils.parser_config(parser.parse_args(), cfg_updata=True)
    if cfg.polyaxon:
        from core.utils.rsync_data import get_polyaxon_dataroot, get_polyaxon_output
        cfg.gpu_id = list(range(len(cfg.gpu_id)))
        cfg.train_data = get_polyaxon_dataroot(dir=cfg.train_data)
        cfg.test_data = get_polyaxon_dataroot(dir=cfg.test_data)
        cfg.work_dir = get_polyaxon_output(cfg.work_dir)
        if isinstance(cfg.class_name, str): cfg.class_name = get_polyaxon_dataroot(cfg.class_name)
    return cfg


if __name__ == "__main__":
    cfg = get_parser()
    train = Trainer(cfg)
    train.run()
