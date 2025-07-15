import torch
import math
import numpy as np
import argparse
import os
from meta import Meta
from Simulations.LOR.LOR_sysmdl import SystemModel

def main(args):
    print(args)

    args.use_cuda = False
    if args.use_cuda:
        if torch.cuda.is_available():
            print("Using GPU")
            device = torch.device('cuda')
        else:
            raise Exception("No GPU found, please set args.use_cuda = False")
    else:
        print("Using CPU")
        device = torch.device('cpu')

    path_data = './MAML_data/LOR/'
    db_train = SystemModel(batchsz=args.task_num,
                              n_way=args.n_way,
                              k_shot=args.k_spt,
                              k_shot_test=args.k_spt_test,
                              q_query=args.q_qry,
                              data_path=path_data,
                              Is_GenData=False,
                              use_cuda=args.use_cuda,
                              Is_linear=False)  #调用就会产生训练和测试数据

    maml = Meta(args, db_train, is_linear_net=True)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    weights = torch.load('./MAML_data/LOR/train/weights.pt')
    weights = torch.tensor(weights, device=device)

    # # Here, we reduce the testing part for accelerating MAML-KalmanNet training
    # for step in range(args.epoch):
    #     state_spt, obs_spt, state_qry, obs_qry, select_num = db_train.next()
    #     state_spt, obs_spt, state_qry, obs_qry = torch.from_numpy(state_spt), torch.from_numpy(obs_spt), \
    #         torch.from_numpy(state_qry), torch.from_numpy(obs_qry)
    #     epoch_weights = weights[select_num]
    #     state_spt, obs_spt, state_qry, obs_qry = state_spt.to(device), obs_spt.to(device), state_qry.to(device), obs_qry.to(device)
    #
    #     if step <= args.epoch / 2:
    #         loss_dB, count_num = maml(state_spt, obs_spt, state_qry, obs_qry, epoch_weights)
    #     else:
    #         loss_dB, count_num = maml.forward_second(state_spt, obs_spt, state_qry, obs_qry)
    #
    #     if step % 1 == 0:
    #         print('step:' + str(step) + ' loss_dB: ' + str(loss_dB) + ' count_num ' + str(count_num))
    #
    #     if step % 20 == 0 and step != 0:
    #         torch.save(maml.base_net.state_dict(), './MAML_model/LOR/basenet_' + str(step) + '.pt')

    # Instantiate Meta and load pretrained weights
    maml = Meta(args, db_train, is_linear_net=True).to(device)
    # 自动加载指定步骤保存的模型（默认使用最后一个step）
    checkpoint_path = './MAML_model/LOR/basenet_200.pt'
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    maml.base_net.load_state_dict(state_dict)
    maml.base_net.eval()

    # # Testing loop
    # total_loss = 0.0
    # total_count = 0
    # state_spt, obs_spt, state_qry, obs_qry, _ = db_train.next(mode='test')
    # # Convert to tensors
    # state_spt = torch.from_numpy(state_spt).to(device)
    # obs_spt = torch.from_numpy(obs_spt).to(device)
    # state_qry = torch.from_numpy(state_qry).to(device)
    # obs_qry = torch.from_numpy(obs_qry).to(device)
    #
    # # Perform one meta-test adaptation (no re-weight)
    # # with torch.no_grad():
    # loss_dB, count_num = maml.forward_test(state_spt, obs_spt, state_qry, obs_qry)
    #
    # total_loss += loss_dB * count_num
    # total_count += count_num
    #
    # avg_loss_dB = total_loss / max(total_count, 1)
    # # print(f"Test completed over {num_tasks} tasks")
    # print(f"Average loss (dB): {avg_loss_dB:.4f}")
    # ive-KNet-ICASSP24-main\simulations\Lorenz_Atractor\data_nonlinear_imm\linear_gaussian_imm
    DatafolderName = 'MAML_data/LOR/test' + '/'
    r2 = torch.tensor([10])
    dataFileName = [f'data_lor_v10_r{r2.item():.3f}_T100_linear_gaussian_f_initial_x0_p0_pretrained_noise-shift.pt']
    [train_input_long, train_target_long, cv_input, cv_target, test_input, test_target, train_init, cv_init,
     test_init] = torch.load(DatafolderName + dataFileName[0], map_location=device)
    # state, obs = test_target[0:60], test_input[0:60] # [batch_size, 3, 100]
    state, obs = test_target, test_input  # [batch_size, 3, 100]
    #readme: 注意 from state_dict_learner import Learner 中line32要修改成test_input的batchsize大小
    mse_arr, mse_avg, mse_dB, x_out, t, mse_time = maml.forward_test(state.to(device), obs.to(device))
    # 1) 转成 torch.Tensor
    mse_arr_t = torch.from_numpy(mse_arr)  # [batch]
    mse_avg_t = torch.tensor(mse_avg)  # scalar
    mse_dB_t = torch.tensor(mse_dB)  # scalar
    x_out_t = torch.from_numpy(x_out)  # [batch, x_dim, seq_len-1]
    t_t = torch.from_numpy(t)  # [seq_len-1]
    mse_time_t = torch.from_numpy(mse_time)  # [batch, seq_len-1]

    # 提取倒数第 3 和第 7 的值
    last_3 = mse_time_t[:, -3]  # [batch]
    last_7 = mse_time_t[:, -7]  # [batch]

    # 填充最后两维
    fill_values = torch.stack([last_3, last_7], dim=1)  # [batch, 2]
    mse_time_extended = torch.cat([mse_time_t, fill_values], dim=1)  # [batch, seq_len]

    # 2) 保存
    save_dict = {
        'mse_arr': mse_arr_t,
        'mse_avg': mse_avg_t,
        'mse_dB': mse_dB_t,
        'x_out': x_out_t,
        't': t_t,
        'mse_time': mse_time_extended,
    }
    filename = f'test_results_r{r2.item():.3f}.pth'
    save_path = './MAML_data/LOR/'
    save_path = os.path.join(save_path, filename)
    torch.save(save_dict, save_path)
    print(f"Saved test results to {save_path}")
    # print(f"Average test MSE(dB): {mse_dB:.4f}")
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=201)
    argparser.add_argument('--n_way', type=int, help='n way', default=8)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=6)  # k_spt>=batch_size
    argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=30)
    argparser.add_argument('--q_qry', type=int, help='q shot for query set', default=15)
    argparser.add_argument('--batch_size', type=int, help='batch size for train MAML', default=4)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.05)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=4)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=6)
    argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=False)

    args = argparser.parse_args()

    main(args)
