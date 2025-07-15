import math
import torch
import numpy as np
from torch import nn
from torch import optim
from state_dict_learner import Learner
from filter import Filter as Filter
import time

class Meta(nn.Module):

    def __init__(self, args, model, is_linear_net=True):
        super(Meta, self).__init__()
        # super().__init__()
        self.args = args
        self.is_linear_net = is_linear_net
        self.update_lr = args.update_lr  # 0.4
        self.meta_lr = args.meta_lr  # 0.001
        self.n_way = args.n_way  # 5
        self.k_spt = args.k_spt  # 1
        self.k_qry = args.q_qry  # 15
        self.task_num = args.task_num  # 32
        self.update_step = args.update_step  # 5
        self.update_step_test = args.update_step_test  # 10
        if args.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.my_filter = Filter(args, model, is_linear_net=is_linear_net)
        self.model = model
        self.base_net = Learner(self.model.x_dim, self.model.y_dim, args, is_linear_net).to(self.device)

        self.meta_optim = optim.Adam(self.base_net.parameters(), lr=self.meta_lr)
        self.loss_fn = torch.nn.MSELoss()
        self.weight_decay = [0.3, 0.3, 0.2, 0.1, 0.1, 0.01]
        self.use_weight = True  # Whether to employ weights to balance the influence of different tasks

    def forward(self, state_spt, obs_spt, state_qry, obs_qry, weights):

        # turn on anomaly detection mode
        # torch.autograd.set_detect_anomaly(True)
        while True:
            num = 0
            restart = False
            task_num = state_spt.shape[0]
            count_num = task_num
            loss_q = 0
            gradients = {}
            temp_gradients = {}
            losses = torch.tensor(0.).to(self.device)
            for name, param in self.base_net.named_parameters():
                if param.requires_grad:
                    gradients[name] = torch.zeros_like(param)
                    temp_gradients[name] = torch.zeros_like(param)

            for i in range(task_num):

                task_model = Learner(self.model.x_dim, self.model.y_dim, self.args, self.is_linear_net).to(self.device)
                task_model.load_state_dict(self.base_net.state_dict())
                task_model.initialize_hidden()
                inner_optimizer = optim.SGD(task_model.parameters(), lr=self.update_lr)

                loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)

                # # 一旦检测到 NaN，就标记并跳出 epoch 循环
                # if torch.isnan(loss):
                #     print(f"[WARNING] Epoch {i} 训练损失为 NaN，重启整个训练流程。")
                #     restart = True
                #     num = num + 1
                #     break

                if math.isnan(loss):
                    count_num = count_num - 1
                    continue

                inner_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                inner_optimizer.step()

                for k in range(1, self.update_step):
                    loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)
                    if math.isnan(loss) or math.isnan(loss_q):
                        count_num = count_num - 1
                        break
                    inner_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                    inner_optimizer.step()

                    loss_q = self.my_filter.compute_x_post_qry(state_qry[i], obs_qry[i], task_net=task_model)
                    loss_q.backward()
                    torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                    for name, param in task_model.named_parameters():
                        if param.grad is not None:
                            if self.use_weight:
                                temp_gradients[name] += param.grad.clone() * self.weight_decay[k-1] * weights[i]
                            else:
                                temp_gradients[name] += param.grad.clone() * self.weight_decay[k-1]

                if math.isnan(loss) or math.isnan(loss_q):
                    for name, param in task_model.named_parameters():
                        if param.grad is not None:
                            temp_gradients[name] = torch.zeros_like(param)
                    continue
                else:
                    for name, param in task_model.named_parameters():
                        if param.grad is not None:
                            gradients[name] += temp_gradients[name].clone()
                            temp_gradients[name] = torch.zeros_like(param)

                losses += loss_q.clone()
                # print("losses=", losses)
                # 一旦检测到 NaN，就标记并跳出 epoch 循环
                if torch.isnan(losses):
                    print(f"[WARNING] Epoch 训练损失为 NaN，重启整个训练流程。")
                    restart = True
                    num = num + 1
                    break

            if count_num == 0:
                # print("count_num =0")
                restart = True
                # break
                # return 0, 0

            for param in self.base_net.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            for name, param in self.base_net.named_parameters():
                if name in gradients:
                    param.grad = gradients[name] / count_num

            torch.nn.utils.clip_grad_norm_(self.base_net.parameters(), 1)
            self.meta_optim.step()

            if restart:
                continue
            else:
                break

        return 10 * torch.log10(losses/count_num), count_num

    def forward_second(self, state_spt, obs_spt, state_qry, obs_qry):

        # torch.autograd.set_detect_anomaly(True)

        task_num = state_spt.shape[0]
        count_num = task_num
        meta_loss = torch.tensor(0.)
        is_qry_nan = False
        gradients = {}
        for name, param in self.base_net.named_parameters():
            if param.requires_grad:
                gradients[name] = torch.zeros_like(param)

        for i in range(task_num):

            task_model = Learner(self.model.x_dim, self.model.y_dim, self.args, self.is_linear_net).to(self.device)
            task_model.load_state_dict(self.base_net.state_dict())
            task_model.initialize_hidden(is_train=True)
            inner_optimizer = optim.Adam(task_model.parameters(), lr=self.meta_lr)

            loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)
            if torch.isnan(loss).item():
                count_num -= 1
                continue

            inner_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
            inner_optimizer.step()

            for k in range(1, self.update_step):

                loss = self.my_filter.compute_x_post(state_spt[i], obs_spt[i], task_net=task_model)
                if torch.isnan(loss).item():
                    count_num -= 1
                    is_qry_nan = True
                    break

                inner_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 1)
                inner_optimizer.step()

            if is_qry_nan:
                is_qry_nan = False
                continue

            task_model.initialize_hidden(is_train=False)
            loss_qry = self.my_filter.compute_x_post_qry(state_qry[i], obs_qry[i], task_net=task_model)

            meta_loss = meta_loss + loss_qry

        if count_num == 0:
            return 0, 0

        meta_loss = meta_loss / task_num
        meta_loss.backward()

        for name, param in task_model.named_parameters():
            if param.grad is not None:
                gradients[name] += param.grad.clone()

        for param in self.base_net.parameters():
            if param.grad is not None:
                param.grad.zero_()

        for name, param in self.base_net.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()

        torch.nn.utils.clip_grad_norm_(self.base_net.parameters(), 1)
        self.meta_optim.step()

        return 10 * torch.log10(meta_loss), count_num

    def forward_test(self, state: torch.Tensor, obs: torch.Tensor):
        """
        Test function for a batch of sequences.

        Args:
            state: 真实状态，shape [batch_size, x_dim, seq_len]
            obs:   观测值，shape [batch_size, y_dim, seq_len]

        Returns:
            [
                MSE_test_linear_arr,  # np.ndarray, shape [batch_size]
                MSE_test_linear_avg,  # float
                MSE_test_dB_avg,      # float
                x_out_test,           # np.ndarray, shape [batch_size, x_dim, seq_len-1]
                t,                    # np.ndarray, shape [seq_len-1]
                MSE_time_avg          # np.ndarray, shape [seq_len-1]
            ]
        """
        self.base_net.eval()
        batch_size, x_dim, seq_len = state.shape
        device = self.device

        # 关掉梯度计算
        with torch.no_grad():
            start = time.time()
            # 用新的 reset_test 来按 batch 大小初始化 Filter
            self.my_filter.reset_test(batch_size)
            # 2) 初始化 Learner (GRU) 隐状态
            self.base_net.initialize_hidden(is_train=False)
            # 逐步滤波
            for k in range(1, seq_len):
                y_k = obs[:, :, k].unsqueeze(-1)                   # [batch, y_dim, 1]
                self.my_filter.filtering(y_k, task_net=self.base_net)
                # self.my_filter.compute_x_post_test(state, obs, task_net=self.base_net)

            end = time.time()
            runtime = end - start

            print("Inference Time:", runtime)
            # 从滤波器中取出整条历史轨迹，丢掉初始值
            # state_history: [batch, x_dim, seq_len]
            x_hist = self.my_filter.state_history[:, :, 2:]         # [batch, x_dim, seq_len-1]

            # 计算误差
            true_traj = state[:, :, 2:]                            # [batch, x_dim, seq_len-1]
            err = x_hist - true_traj                               # [batch, x_dim, seq_len-1]
            # 每时间维度上的 MSE 曲线
            MSE_time = err.pow(2).mean(dim=1)                      # [batch, seq_len-1]
            # 每条序列的平均 MSE（线性尺度）
            MSE_test_linear_arr = MSE_time.mean(dim=1).cpu().numpy()    # [batch]
            # 平均 MSE（线性尺度与 dB）
            MSE_test_linear_avg = float(MSE_test_linear_arr.mean())
            MSE_test_dB_avg    = 10.0 * np.log10(MSE_test_linear_avg)

            # Standard deviation

            MSE_test_linear_std = np.std(MSE_test_linear_arr)

            # Confidence interval
            test_std_dB = 10 * np.log10(
                MSE_test_linear_std + MSE_test_linear_avg) - MSE_test_dB_avg

            # 时间步索引
            t = np.arange(1, seq_len)
            # 按时间步对 MSE 求批平均
            # MSE_time_avg = MSE_time.mean(dim=0).cpu().numpy()           # [seq_len-1]
            MSE_time_avg = MSE_time.cpu().numpy()  # [batch, seq_len-1]

            print("MAML_KalmanNet MSE Test in dB:", MSE_test_dB_avg)
            print("MAML_KalmanNet STD Test in dB:", test_std_dB)
        return [
            MSE_test_linear_arr,
            MSE_test_linear_avg,
            MSE_test_dB_avg,
            x_hist.cpu().numpy(),
            t,
            MSE_time_avg,
        ]