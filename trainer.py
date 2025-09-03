import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_sparse import transpose
from torch_geometric.utils import is_undirected, sort_edge_index
from collections import defaultdict
import numpy as np
from datetime import datetime
import random
from itertools import combinations
import copy

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score


class BaseTrainer(object):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        self.method_name = None
        self.checkpoints_path = None

        self.model = model
        self.explainer = explainer
        self.dataloader = dataloader
        self.cfg = cfg

        self.device = device

        self.best_valid_score = 0.0
        self.lowest_valid_loss = float('inf')
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(os.path.join(save_dir, 'checkpoints'))

    def set_method_name(self, method_name):
        self.method_name = method_name
        self.checkpoints_path = os.path.join(self.save_dir, 'checkpoints',
                                             f'{self.method_name}_{self.cfg.dataset_name}.pth')

    def _train_batch(self, data):
        raise NotImplementedError

    def _valid_batch(self, data):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def valid(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    @staticmethod
    def process_data(data, use_edge_attr):
        if not use_edge_attr:
            data.edge_attr = None
        if data.get('edge_label', None) is None:
            data.edge_label = torch.zeros(data.edge_index.shape[1])
        return data

    @torch.inference_mode()
    def calculate_shd_auc_fid_acc(self, method_name, ensemble_numbers=[0]):
        assert self.cfg.multi_label is False  # only support binary classification now
        assert len(ensemble_numbers) % 2 == 0

        ori_data = []
        for data in self.dataloader['test_by_sample']:
            ori_data.append(copy.deepcopy(data))

        for model_index in ensemble_numbers:
            new_checkpoints_path = f'{self.checkpoints_path[:-4]}_{model_index}.pth'  # att_ba_2motifs.pth -> att_ba_2motifs_0.pth
            self.load_model(new_checkpoints_path)
            self.model.eval()
            self.explainer.eval()
            for data_index, data in enumerate(self.dataloader['test_by_sample']):
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                         batch=data.batch)
                att = self.concrete_sample(self.explainer(emb, data.edge_index, data.batch), training=False)
                edge_att = self.process_att_to_edge_att(data, att)
                ori_data[data_index][f'edge_att_{model_index}'] = edge_att.squeeze()  # save generated exp for auc

                minus_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index,
                                               edge_attr=data.edge_attr, batch=data.batch, edge_atten=edge_att)
                minus_att = self.concrete_sample(self.explainer(minus_emb, data.edge_index, data.batch), training=False)
                minus_edge_att = self.process_att_to_edge_att(data, minus_att)
                if 'cal' in method_name:
                    s_edge_att = 1 - edge_att
                    c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch,
                                               edge_atten=edge_att)
                    s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch,
                                               edge_atten=s_edge_att)
                    csi_emb = s_emb + c_emb
                    logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)  # [1, 1]
                    '''for fid-'''
                    minus_s_edge_att = 1 - minus_edge_att
                    minus_c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                     batch=data.batch,
                                                     edge_atten=minus_edge_att)
                    minus_s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                     batch=data.batch,
                                                     edge_atten=minus_s_edge_att)
                    minus_csi_emb = minus_s_emb + minus_c_emb
                    logits_minus = self.model.get_pred_from_csi_emb(emb=minus_csi_emb, batch=data.batch)
                else:
                    logits = self.model(x=data.x, edge_index=data.edge_index,
                                        edge_attr=data.edge_attr, batch=data.batch, edge_atten=edge_att)
                    '''for fid-'''
                    logits_minus = self.model(x=data.x, edge_index=data.edge_index,
                                              edge_attr=data.edge_attr, batch=data.batch, edge_atten=minus_edge_att)

                ori_data[data_index][f'y_hat_{model_index}'] = logits.sigmoid()  # binary classification
                ori_data[data_index][f'y_hat_minus_{model_index}'] = logits_minus.sigmoid()  # binary classification

        '''calculate fid'''
        fid_scores = []
        for model_index in range(len(ensemble_numbers)):
            fid_model_scores = []
            for data in ori_data:
                logits = data[f'y_hat_{model_index}'].squeeze()
                pred_ori = (logits > 0.5).int()
                logits_minus = data[f'y_hat_minus_{model_index}'].squeeze()
                pred_minus = (logits_minus > 0.5).int()
                score = torch.abs(pred_ori - pred_minus).item()
                fid_model_scores.append(score)
            fid_scores.append(np.mean(fid_model_scores))
        fid_mean, fid_std = np.mean(fid_scores), np.std(fid_scores)

        '''calculate shd (before & after EE)'''
        # before EE
        n = len(ensemble_numbers)
        all_scores_dict = defaultdict(list)
        for data in ori_data:
            edge_atts = [data[f'edge_att_{i}'] for i in range(n)]
            combinations_list = list(combinations(range(n), 1))
            for pair in combinations(combinations_list, 2):
                if set(pair[0]).isdisjoint(pair[1]):
                    edge_atts_first = torch.stack([edge_atts[i] for i in pair[0]]).mean(dim=0)  # [edge_num]
                    edge_atts_second = torch.stack([edge_atts[i] for i in pair[1]]).mean(dim=0)
                    mae_distance = torch.abs(edge_atts_first - edge_atts_second).mean().item()
                    score = mae_distance
                    all_scores_dict[f'{pair[0]}_{pair[1]}'].append(score)
        shd_ori_scores = []
        for key, item in all_scores_dict.items():
            shd_ori_scores.append(np.mean(item))
        shd_ori_mean, shd_ori_std = np.mean(shd_ori_scores), np.std(shd_ori_scores)

        # after EE
        all_scores_dict = defaultdict(list)
        for data in ori_data:
            edge_atts = [data[f'edge_att_{i}'] for i in range(n)]
            combinations_list = list(combinations(range(n), int(n / 2)))
            for pair in combinations(combinations_list, 2):
                if set(pair[0]).isdisjoint(pair[1]):
                    edge_atts_first = torch.stack([edge_atts[i] for i in pair[0]]).mean(dim=0)  # [edge_num]
                    edge_atts_second = torch.stack([edge_atts[i] for i in pair[1]]).mean(dim=0)
                    mae_distance = torch.abs(edge_atts_first - edge_atts_second).mean().item()
                    score = mae_distance
                    all_scores_dict[f'{pair[0]}_{pair[1]}'].append(score)
        shd_ee_scores = []
        for key, item in all_scores_dict.items():
            shd_ee_scores.append(np.mean(item))
        shd_ee_mean, shd_ee_std = np.mean(shd_ee_scores), np.std(shd_ee_scores)

        '''calculate auc'''
        # before EE
        model_auc_list = []
        # all_scores_dict = defaultdict(list)  # paper figure (instance visualization)
        for model_index in range(n):
            edge_att_list = []
            exp_label_list = []
            for data in ori_data:
                edge_att = data[f'edge_att_{model_index}']
                exp_label = data.edge_label.data
                edge_att_list.append(edge_att)
                exp_label_list.append(exp_label)
                # score = roc_auc_score(exp_label.cpu().numpy(), edge_att.cpu().numpy())  # for instance visualization
                # all_scores_dict[f'{model_index}'].append(score)
            model_auc = roc_auc_score(torch.cat(exp_label_list).cpu(), torch.cat(edge_att_list).cpu())
            model_auc_list.append(model_auc)
        auc_ori_mean, auc_ori_std = np.mean(model_auc_list), np.std(model_auc_list)

        # after EE
        model_auc_list = []
        # all_scores_dict = defaultdict(list)  # paper figure (instance visualization)
        combinations_list = list(combinations(range(n), int(n / 2)))
        for pair in combinations_list:
            edge_att_list = []
            exp_label_list = []
            for data in ori_data:
                edge_atts = [data[f'edge_att_{i}'] for i in range(n)]
                edge_att = torch.stack([edge_atts[i] for i in pair]).mean(dim=0)
                exp_label = data.edge_label.data
                edge_att_list.append(edge_att)
                exp_label_list.append(exp_label)
                # score = roc_auc_score(exp_label.cpu().numpy(), edge_att.cpu().numpy())  # for instance visualization
                # all_scores_dict[f'{pair}'].append(score)
            model_auc = roc_auc_score(torch.cat(exp_label_list).cpu(), torch.cat(edge_att_list).cpu())
            model_auc_list.append(model_auc)
        auc_ee_mean, auc_ee_std = np.mean(model_auc_list), np.std(model_auc_list)

        '''acc'''
        # before EE
        model_acc_list = []
        for model_index in range(n):
            y_hat_list = []
            y_list = []
            for data in ori_data:
                y_hat = data[f'y_hat_{model_index}'].squeeze().item()
                y = data.y.squeeze().item()
                y_hat_list.append(1 if y_hat > 0.5 else 0)
                y_list.append(1 if y > 0.5 else 0)
            acc = accuracy_score(y_list, y_hat_list)
            model_acc_list.append(acc)
        acc_ori_mean, acc_ori_std = np.mean(model_acc_list), np.std(model_acc_list)
        # after EE
        model_acc_list = []
        combinations_list = list(combinations(range(n), int(n / 2)))
        for pair in combinations_list:
            y_hat_list = []
            y_list = []
            for data in ori_data:
                y_hat = torch.stack([data[f'y_hat_{i}'] for i in pair]).mean(dim=0).squeeze().item()
                y = data.y.squeeze().item()
                y_hat_list.append(1 if y_hat > 0.5 else 0)
                y_list.append(1 if y > 0.5 else 0)
            acc = accuracy_score(y_list, y_hat_list)
            model_acc_list.append(acc)
        acc_ee_mean, acc_ee_std = np.mean(model_acc_list), np.std(model_acc_list)

        print("====================================================================================")
        print("before ensemble:")
        print(f"shd: {shd_ori_mean:.4f}±{shd_ori_std:.4f}\n"
              f"auc: {auc_ori_mean:.4f}±{auc_ori_std:.4f}\n"
              f"acc: {acc_ori_mean:.4f}±{acc_ori_std:.4f}\n"
              f"fid-: {fid_mean:.4f}±{fid_std:.4f}")
        print("------------------------------------------------------------------------------------")
        print("after ensemble (5 models):")
        print(f"shd: {shd_ee_mean:.4f}±{shd_ee_std:.4f}\n"
              f"auc: {auc_ee_mean:.4f}±{auc_ee_std:.4f}\n"
              f"acc: {acc_ee_mean:.4f}±{acc_ee_std:.4f}")
        print("====================================================================================")


class ATTTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, logits, labels):
        ce_loss = self.criterion(logits, labels)
        loss = ce_loss * self.ce_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class ATTCDTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef
        self.cd_loss_coef = cfg.cd_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

        self.ee_models = []
        self.ee_explainers = []

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, logits, labels, att_cd, att_ee):
        ce_loss = self.criterion(logits, labels)
        distill_loss = torch.mean(torch.abs(att_cd - att_ee))
        loss = ce_loss * self.ce_loss_coef + distill_loss * self.cd_loss_coef
        return loss, distill_loss * self.cd_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)
        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(logits=clf_logits, labels=data.y, att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)
        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(logits=clf_logits, labels=data.y, att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/archived_checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        for model_index in range(5):  # 5 ensemble models
            path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{model_index}.pth'
            new_model = copy.deepcopy(self.model)
            new_explainer = copy.deepcopy(self.explainer)
            state = torch.load(path, map_location=self.device)
            new_model.load_state_dict(state['model_state_dict'])
            new_explainer.load_state_dict(state['explainer_state_dict'])
            new_model.eval()
            new_explainer.eval()
            self.ee_models.append(new_model)
            self.ee_explainers.append(new_explainer)

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)

        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()

        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class SIZETrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef
        self.sparsity_mask_coef = cfg.sparsity_mask_coef
        self.sparsity_ent_coef = cfg.sparsity_ent_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    def sparsity(self, edge_mask, eps=1e-6):
        sparsity = 0.
        # sparsity += self.sparsity_mask_coef * edge_mask.mean()
        sparsity += self.c * edge_mask.mean()
        # ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        # sparsity += self.sparsity_ent_coef * ent.mean()
        return sparsity

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_sparsity_c(self, current_epoch):
        c = self.sparsity_mask_coef * (current_epoch + 1) / 10
        if c > self.sparsity_mask_coef:
            c = self.sparsity_mask_coef
        return c

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels):
        ce_loss = self.criterion(logits, labels)
        reg_loss = self.sparsity(att)
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.c = self.get_sparsity_c(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (self.c == self.sparsity_mask_coef) and (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class SIZECDTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef
        self.sparsity_mask_coef = cfg.sparsity_mask_coef
        self.sparsity_ent_coef = cfg.sparsity_ent_coef
        self.cd_loss_coef = cfg.cd_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

        self.ee_models = []
        self.ee_explainers = []

    def sparsity(self, edge_mask, eps=1e-6):
        sparsity = 0.
        # sparsity += self.sparsity_mask_coef * edge_mask.mean()
        sparsity += self.c * edge_mask.mean()
        # ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        # sparsity += self.sparsity_ent_coef * ent.mean()
        return sparsity

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_sparsity_c(self, current_epoch):
        c = self.sparsity_mask_coef
        return c

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels, att_cd, att_ee):
        ce_loss = self.criterion(logits, labels)
        reg_loss = self.sparsity(att)
        distill_loss = torch.mean(torch.abs(att_cd - att_ee))
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef + distill_loss * self.cd_loss_coef
        return loss, distill_loss * self.cd_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)
        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)
        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/archived_checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        for model_index in range(5):
            path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{model_index}.pth'
            new_model = copy.deepcopy(self.model)
            new_explainer = copy.deepcopy(self.explainer)
            state = torch.load(path, map_location=self.device)
            new_model.load_state_dict(state['model_state_dict'])
            new_explainer.load_state_dict(state['explainer_state_dict'])
            new_model.eval()
            new_explainer.eval()
            self.ee_models.append(new_model)
            self.ee_explainers.append(new_explainer)

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            self.c = self.get_sparsity_c(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class GSATTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels):
        ce_loss = self.criterion(logits, labels)
        reg_loss = (att * torch.log(att / self.r + 1e-6)
                    + (1 - att) * torch.log((1 - att) / (1 - self.r + 1e-6) + 1e-6)).mean()
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (self.r == self.final_r) and (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class GSATCDTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef
        self.cd_loss_coef = cfg.cd_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

        self.ee_models = []
        self.ee_explainers = []

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        r = self.final_r
        return r

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels, att_cd, att_ee):
        ce_loss = self.criterion(logits, labels)
        reg_loss = (att * torch.log(att / self.r + 1e-6)
                    + (1 - att) * torch.log((1 - att) / (1 - self.r + 1e-6) + 1e-6)).mean()
        distill_loss = torch.mean(torch.abs(att_cd - att_ee))
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef + distill_loss * self.cd_loss_coef
        return loss, distill_loss * self.cd_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)

        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)
        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(att=att, logits=clf_logits, labels=data.y, att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/archived_checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        for model_index in range(5):
            path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{model_index}.pth'
            new_model = copy.deepcopy(self.model)
            new_explainer = copy.deepcopy(self.explainer)
            state = torch.load(path, map_location=self.device)
            new_model.load_state_dict(state['model_state_dict'])
            new_explainer.load_state_dict(state['explainer_state_dict'])
            new_model.eval()
            new_explainer.eval()
            self.ee_models.append(new_model)
            self.ee_explainers.append(new_explainer)

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")

            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class CALTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att

        self.csi_loss_coef = cfg.csi_loss_coef
        self.c_loss_coef = cfg.c_loss_coef
        self.s_loss_coef = cfg.s_loss_coef

        self.num_class = cfg.num_class

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, c_logits, s_logits, csi_logits, labels):
        if self.num_class == 2:
            s_prob = torch.sigmoid(s_logits)
            s_loss = (s_prob * torch.log(s_prob / 0.5 + 1e-6)
                      + (1 - s_prob) * torch.log((1 - s_prob) / (1 - 0.5 + 1e-6) + 1e-6)).mean()
        else:
            log_s_probs = torch.log_softmax(s_logits, dim=-1)
            uniform_target = torch.ones_like(s_logits, dtype=torch.float).cuda() / self.num_class
            s_loss = F.kl_div(log_s_probs, uniform_target, reduction='batchmean')

        c_loss = self.criterion(c_logits, labels)
        csi_loss = self.criterion(csi_logits, labels)
        loss = csi_loss * self.csi_loss_coef + c_loss * self.c_loss_coef + s_loss * self.s_loss_coef

        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)

        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()

        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class CALCDTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att

        self.csi_loss_coef = cfg.csi_loss_coef
        self.c_loss_coef = cfg.c_loss_coef
        self.s_loss_coef = cfg.s_loss_coef
        self.cd_loss_coef = cfg.cd_loss_coef

        self.num_class = cfg.num_class

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

        self.ee_models = []
        self.ee_explainers = []

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, c_logits, s_logits, csi_logits, labels, att_cd, att_ee):
        if self.num_class == 2:
            s_prob = torch.sigmoid(s_logits)
            s_loss = (s_prob * torch.log(s_prob / 0.5 + 1e-6)
                      + (1 - s_prob) * torch.log((1 - s_prob) / (1 - 0.5 + 1e-6) + 1e-6)).mean()
        else:
            log_s_probs = torch.log_softmax(s_logits, dim=-1)
            uniform_target = torch.ones_like(s_logits, dtype=torch.float).cuda() / self.num_class
            s_loss = F.kl_div(log_s_probs, uniform_target, reduction='batchmean')

        c_loss = self.criterion(c_logits, labels)
        csi_loss = self.criterion(csi_logits, labels)
        distill_loss = torch.mean(torch.abs(att_cd - att_ee))
        loss = csi_loss * self.csi_loss_coef + c_loss * self.c_loss_coef + s_loss * self.s_loss_coef + distill_loss * self.cd_loss_coef

        return loss, distill_loss * self.cd_loss_coef

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)

        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)
        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                                           labels=data.y,
                                           att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()

        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        emb_no_grad = emb.detach()
        att_log_logit_cd = self.explainer(emb_no_grad, data.edge_index, data.batch)
        att_cd = self.concrete_sample(att_log_logit_cd, training=False)
        edge_att_cd = self.process_att_to_edge_att(data, att_cd)

        '''ee result'''
        all_ee_edge_atts = []
        for i in range(len(self.ee_models)):
            emb_i = self.ee_models[i].get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                              batch=data.batch)
            att_log_logit_i = self.ee_explainers[i](emb_i, data.edge_index, data.batch)
            att_i = self.concrete_sample(att_log_logit_i, training=False)
            edge_att_i = self.process_att_to_edge_att(data, att_i)
            all_ee_edge_atts.append(edge_att_i)
        all_ee_edge_atts = torch.stack(all_ee_edge_atts, dim=0).detach()
        edge_att_ee = all_ee_edge_atts.mean(dim=0)

        loss, distill_loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits,
                                           labels=data.y,
                                           att_cd=edge_att_cd,
                                           att_ee=edge_att_ee)

        info = dict()
        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return distill_loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train_ft(self, cur_index):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        path = f'./outputs/archived_checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{cur_index}.pth'
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])

        for model_index in range(5):
            path = f'./outputs/checkpoints/{self.method_name[:-3]}_{self.cfg.dataset_name}_{model_index}.pth'
            new_model = copy.deepcopy(self.model)
            new_explainer = copy.deepcopy(self.explainer)
            state = torch.load(path, map_location=self.device)
            new_model.load_state_dict(state['model_state_dict'])
            new_explainer.load_state_dict(state['explainer_state_dict'])
            new_model.eval()
            new_explainer.eval()
            self.ee_models.append(new_model)
            self.ee_explainers.append(new_explainer)

        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()

            for conv in self.model.convs:
                for param in conv.parameters():
                    param.requires_grad = False
                conv.eval()

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (valid_metrics['acc'] > self.best_valid_score) or (
                    valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss):
                self.save_model(self.checkpoints_path)
            self.best_valid_score = valid_metrics['acc']
            self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []
        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)
            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


def get_trainer(method_name, model, explainer, dataloader, cfg, device, save_dir):
    trainer = None
    if method_name == 'att':
        trainer = ATTTrainer(model=model,
                             explainer=explainer,
                             dataloader=dataloader,
                             cfg=cfg,
                             device=device,
                             save_dir=save_dir)
    elif method_name == 'att_cd':
        trainer = ATTCDTrainer(model=model,
                               explainer=explainer,
                               dataloader=dataloader,
                               cfg=cfg,
                               device=device,
                               save_dir=save_dir)
    elif method_name == 'size':
        trainer = SIZETrainer(model=model,
                              explainer=explainer,
                              dataloader=dataloader,
                              cfg=cfg,
                              device=device,
                              save_dir=save_dir)
    elif method_name == 'size_cd':
        trainer = SIZECDTrainer(model=model,
                                explainer=explainer,
                                dataloader=dataloader,
                                cfg=cfg,
                                device=device,
                                save_dir=save_dir)
    elif method_name == 'gsat':
        trainer = GSATTrainer(model=model,
                              explainer=explainer,
                              dataloader=dataloader,
                              cfg=cfg,
                              device=device,
                              save_dir=save_dir)
    elif method_name == 'gsat_cd':
        trainer = GSATCDTrainer(model=model,
                                explainer=explainer,
                                dataloader=dataloader,
                                cfg=cfg,
                                device=device,
                                save_dir=save_dir)
    elif method_name == 'cal':
        trainer = CALTrainer(model=model,
                             explainer=explainer,
                             dataloader=dataloader,
                             cfg=cfg,
                             device=device,
                             save_dir=save_dir)
    elif method_name == 'cal_cd':
        trainer = CALCDTrainer(model=model,
                               explainer=explainer,
                               dataloader=dataloader,
                               cfg=cfg,
                               device=device,
                               save_dir=save_dir)
    trainer.set_method_name(method_name)
    return trainer
