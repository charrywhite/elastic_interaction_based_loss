import os
import torch
import torch.optim as optim

from functools import partial
from argparse import ArgumentParser

from unet.unet import UNet2D
from unet.model import Model
from unet.utils import MetricList
from unet.metrics import jaccard_index, f1_score, LogNLLLoss
from unet.dataset import JointTransform2D, ImageToImage2D, Image2D
from unet.elastic_loss import EnergyLoss

parser = ArgumentParser()
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--test_dataset', required=True,type=str)
parser.add_argument('--save_path', required=True, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--in_channels', default=3, type=int)
parser.add_argument('--out_channels', default=2, type=int)
parser.add_argument('--depth', default=6, type=int)
parser.add_argument('--alpha', default=0.35, type=float)
parser.add_argument('--sigma', default=0.25, type=float)
parser.add_argument('--width', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--save_model', default=0, type=int)
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--crop', type=int, default=512)
args = parser.parse_args()

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

# tf_train = JointTransform2D(crop=crop)
# tf_val = JointTransform2D(crop=crop)
dataset =  ImageToImage2D(dataset_path = args.train_dataset)
test_dataset = ImageToImage2D(dataset_path = args.test_dataset)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print(f'Train dataset size is {len(train_dataset)},  val dataset size is {len(val_dataset)}, test dataset size is {len(test_dataset)}')


conv_depths = [int(args.width*(2**k)) for k in range(args.depth)]
unet = UNet2D(args.in_channels, args.out_channels, conv_depths)
loss = EnergyLoss(alpha=args.alpha) # use Elastic Loss here
# loss = LogNLLLoss()
optimizer = optim.Adam(unet.parameters(), lr=args.learning_rate)

results_folder = os.path.join(args.save_path, args.model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

metric_list = MetricList({'jaccard': partial(jaccard_index),
                          'f1': partial(f1_score)})

model = Model(unet, loss, optimizer, results_folder, device=args.device)

model.fit_dataset(train_dataset, n_epochs=args.epochs, n_batch=args.batch_size,
                  shuffle=True, val_dataset=val_dataset, save_freq=args.save_freq,
                  save_model=args.save_model, predict_dataset=test_dataset,
                  metric_list=metric_list, verbose=True)
