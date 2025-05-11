import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from IPython.display import display
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image
from dataset.utils import get_transform
from dataset.rasterize import rasterize_sketch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(model, args):
    model.to(device)
    model.eval()
    coordinate_path = os.path.join(args.root_dir, args.dataset_name, args.dataset_name + '_Coordinate')
    with open(coordinate_path, 'rb') as f:
        coordinate = pickle.load(f)
    
    test_sketch = [x for x in coordinate if 'test' in x]
    test_transform = get_transform('test', size=256)
    
    sketch_path = test_sketch[0] 
    vector_x = coordinate[sketch_path]
    sketch_img = rasterize_sketch(vector_x)
    sketch_raw_img = Image.fromarray(sketch_img).convert("RGB")
    sketch_img = test_transform(sketch_raw_img)
    
    with torch.no_grad():
        output_tensor = model(sketch_img.unsqueeze(0).to(device))
        
    output_image = output_tensor.squeeze().cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))
    output_image = np.clip(output_image, 0, 1)
    plt.imsave("output.png", output_image)
    
def evaluate(model, test_dataloader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_dataloader)):
            labels, inputs  = batch["sketch_img"].to(device), batch["positive_img"].to(device), 

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    return test_loss

def training_unet(model, train_dataloader, test_dataloader, args):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1} / {args.epochs}")
        epoch_loss = 0
        for _, batch in enumerate(tqdm(train_dataloader)):
            targets, inputs = batch["sketch_img"].to(device), batch["positive_img"].to(device), 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_loss = epoch_loss / len(train_dataloader)
        test_loss = evaluate(model, test_dataloader, criterion)
        
        if train_loss < best_loss:
            name = "best_" + str(args.dataset_name) + "_unet_model.pth"
            torch.save(model.state_dict(), name)
        torch.save(model.state_dict(), "last_" + str(args.dataset_name) + "_unet_model.pth")
        print(f"Trainning loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        inference(model, args)
        
    return train_losses, test_losses
