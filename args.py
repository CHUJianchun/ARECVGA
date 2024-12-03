import argparse
import torch
# H, B, C, N, O, F, P, S


cmd_opt = argparse.ArgumentParser(description='Args parser')

cmd_opt.add_argument('--batch_size', default=100)
cmd_opt.add_argument('--num_workers', default=32)
cmd_opt.add_argument('--charge_power', default=5)

cmd_opt.add_argument('--charge_scale', default=40)
cmd_opt.add_argument('--hidden_feature', default=16)
cmd_opt.add_argument('--latent_dim', default=16)
cmd_opt.add_argument('--n_layers', default=7)

cmd_opt.add_argument('--attention', default=True)
cmd_opt.add_argument('--node_attr', default=True)

cmd_opt.add_argument('--log_interval', default=5000)
cmd_opt.add_argument('--test_interval', default=1)
cmd_opt.add_argument('--epochs', default=500)

cmd_opt.add_argument('--lr', default=5e-5)
cmd_opt.add_argument('--aelr', default=1e-3)
cmd_opt.add_argument('--genlr', default=5e-7)
cmd_opt.add_argument('--crilr', default=5e-7)
cmd_opt.add_argument('--prelr', default=1e-2)

cmd_opt.add_argument('--genn', default=2)
cmd_opt.add_argument('--crin', default=2)

cmd_opt.add_argument('--weight_decay', default=1e-6)
cmd_opt.add_argument('--gain', default=0.7)
cmd_opt.add_argument('--gain2', default=0.7)
cmd_opt.add_argument('--gain3', default=0.7)
cmd_opt.add_argument('--qm_padding_dim', default=9)
cmd_opt.add_argument('--zinc_padding_dim', default=18)
cmd_opt.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
args, _ = cmd_opt.parse_known_args()
