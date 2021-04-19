from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data_path',  help='Origin image path')
parser.add_argument('--weightsave_path', type=str, default='weight',  help='Origin image path')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--lr_warmup', type=str, default=False, help='Learning Rate warmup scheduler')
parser.add_argument('--num_workers', type=int, default=4, help='Number of threads for data loader, for window set to 0')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=16, help='Validation batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument("--experiment_name", type=str, default=None,help='name of experiment')
parser.add_argument("--lossfn", type=str, default='ce', help="[ce,dicece,boundaryce,cldicece]")
parser.add_argument("--net", type=str, default='attUnet', help="Networks, see models.py")
parser.add_argument("--net_inputch", type=int, default=1, help='dimension of input channel')
parser.add_argument("--net_outputch", type=int, default=2, help='dimension of output channel')        
parser.add_argument("--net_pretrained", type=str, default=None, help='path to weights,')
parser.add_argument("--net_mcdropout", type=float, default=0.0, help='MC dropout rate of bottleneck,')
parser.add_argument("--net_norm", type=str, default='ws', help=['batch','instance','group','ws'])
parser.add_argument("--net_nnblock", type=bool, default=True, help=['True','False'])

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
