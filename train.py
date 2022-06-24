from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torchvision
from torchvision.transforms import transforms
from omegaconf import OmegaConf
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from data import FlickerDataset, get_dataloader, collate_fn, save_checkpoint
from model import EncoderCNN, DecoderRNN, CNNtoRNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_fn(model, criterion, optimizer, train_dataloader, epochs):
    
    for name, param in model.encoder.inception.named_parameters():
        if "fc.weight" or "fc.bias" in name:
            param.requires_grad = True
        
        else:
            param.requires_grad = False
            
    writer = SummaryWriter("runs/Flicker")
    total_step = len(train_dataloader)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        
        for idx,(images, captions) in tqdm(enumerate(train_dataloader)):

            images = images.to(device)
        
            captions = captions.to(device)
            
            outputs = model(images, captions[:-1])
            
            outputs = outputs.permute(0,2,1)

            # outputs shape :: [seq length, vocab size, batch size]
            # captions shape :: [seq length, batch size]

            loss = criterion(outputs, captions)

            epoch_loss += images.shape[0]* loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        epoch_avgloss = epoch_loss / len(train_dataloader.dataset)

        writer.add_scalar("Training_loss", epoch_avgloss, global_step=epoch)

        if epoch % 2 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, epochs, epoch_avgloss))

        
        torch.save(model, 'model.pt')

            


if __name__ == '__main__':

    root_dir = '/home/boltzmann/space/Sanketh/Image-Captioning/Images'
    captions_file = '/home/boltzmann/space/Sanketh/Image-Captioning/captions.csv'

    transform = transforms.Compose([
        transforms.Resize((255,255)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    dataset = FlickerDataset(root_dir=root_dir, captions_file=captions_file, transform=transform)
    vocab = torch.load("vocab.pt")

    pad_idx = vocab.stoi['<PAD>']
 
    train_dataset, valid_dataset = train_test_split(dataset, test_size = 0.2, random_state=0)

    train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn(pad_idx=pad_idx))
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn(pad_idx=pad_idx))

    emb_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1
    learning_rate = 0.001
    epochs = 10
            
    model = CNNtoRNN(emb_size, hidden_size, vocab_size, num_layers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    
    train_fn(model,criterion, optimizer, train_dataloader, epochs)

            
            
            
    