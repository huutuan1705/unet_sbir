import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Unet Fine-Grained SBIR model')
    parsers.add_argument('--is_train_unet', type=bool, default=True)
    
    parsers.add_argument('--dataset_name', type=str, default='ChairV2', help="ChairV2/ShoeV2")
    parsers.add_argument('--backbone_name', type=str, default='InceptionV3', help='VGG16/InceptionV3/ResNet50')
    parsers.add_argument('--root_dir', type=str, default='./../')
    
    parsers.add_argument('--lr', type=float, default=0.001)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--batch_size', type=int, default=48)