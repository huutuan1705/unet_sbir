import torch
import argparse
import torch.utils.data as data
 
from dataset.dataset import FGSBIR_Dataset
from unet.unet import UNet
from unet.train import training_unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Unet Fine-Grained SBIR model')
    parsers.add_argument('--is_train_unet', type=bool, default=True)
    parsers.add_argument('--n_channels', type=int, default=3)
    parsers.add_argument('--n_classes', type=int, default=3)
    
    parsers.add_argument('--dataset_name', type=str, default='ChairV2', help="ChairV2/ShoeV2")
    parsers.add_argument('--backbone_name', type=str, default='InceptionV3', help='VGG16/InceptionV3/ResNet50')
    parsers.add_argument('--root_dir', type=str, default='./../')
    
    parsers.add_argument('--lr', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--batch_size', type=int, default=48)
    
    args = parsers.parse_args()
    dataloader_train, dataloader_test = get_dataloader(args=args)
    
    if args.is_train_unet:
        model_unet = UNet(n_channels=args.n_channels, n_classes=args.n_classes)
        train_losses, test_losses = training_unet(model_unet, dataloader_train, dataloader_test, args)