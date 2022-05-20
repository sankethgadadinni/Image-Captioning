import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
from torchvision.transforms import transforms

from omegaconf import OmegaConf
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from data import FlickerDataset, get_dataloader, collate_fn
from model import EncoderCNN, DecoderRNN, CNNtoRNN


def train_fn(root_dir, annotation_file):
    
    
    transform = transforms.Compose([
        transforms.Resize((356,356)),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_dataloader, dataset = get_dataloader(
            root_dir = root_dir,
            annotation_file = annotation_file,
            transform = transform,
            batch_size = 2056)

    emb_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 0.001
    epochs = 5
    
    writer = SummaryWriter("runs/Flicker")
    step = 0
    
    model = CNNtoRNN(emb_size, hidden_size, vocab_size, num_layers)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    
    for name, param in model.encoder.inception.named_parameters():
        if "fc.weight" or "fc.bias" in name:
            param.requires_grad = True
        
        else:
            param.requires_grad = False
            
    
    for epoch in range(epochs):
        
        for idx,(images, captions) in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
            images = images
            captions = captions
            
            outputs = model(images, captions[:-1])
            
            outputs = outputs.reshape(-1, outputs.shape[2])
            loss = criterion(outputs, captions.reshape(-1))
            
            writer.add_scalar("Training_loss", loss.item(), global_step=step)
            step +=1
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            


if __name__ == '__main__':
    
    train_fn("/home/boltzmann/space/Sanketh/Image-Captioning/flickr8k/images","/home/boltzmann/space/Sanketh/Image-Captioning/flickr8k/captions.txt")
            
            
            
    