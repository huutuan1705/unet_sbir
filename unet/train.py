import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_dataloader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_dataloader)):
            inputs, labels = batch["sketch_img"].to(device), batch["positive_img"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    return test_loss

def training_unet(model, train_dataloader, test_dataloader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    train_losses = []
    test_losses = []
    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1} / {args.epochs}")
        epoch_loss = 0
        for _, batch in enumerate(tqdm(train_dataloader)):
            inputs, targets = batch["sketch_img"].to(device), batch["positive_img"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_dataloader)
        test_loss = evaluate(model, test_dataloader, criterion)
        print("Trainning loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
    return train_losses, test_losses