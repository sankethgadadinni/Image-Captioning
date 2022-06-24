from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torchvision
from PIL import Image
from torchvision.transforms import transforms

from data import FlickerDataset, get_dataloader, collate_fn, save_checkpoint
from model import EncoderCNN, DecoderRNN, CNNtoRNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def evaluate(model, image_path, vocab, device):

    model.eval()
    image = Image.open(image_path).convert("RGB")
        
    transform = transforms.Compose([
        transforms.Resize((255,255)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    image = transform(image)

    output = model.caption_image(image, vocab, device)

    return output



if __name__ == '__main__':

    vocab_path = '/home/boltzmann/space/Sanketh/Image-Captioning/vocab.pt'
    model_path = '/home/boltzmann/space/Sanketh/Image-Captioning/model.pt'
    image_path = '/home/boltzmann/space/Sanketh/Image-Captioning/girl.jpeg'

    
    transform = transforms.Compose([
        transforms.Resize((255,255)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    vocab = torch.load(vocab_path)

    emb_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    num_layers = 1
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNtoRNN(emb_size, hidden_size, vocab_size, num_layers)
    model = model.to(device)

    model = torch.load(model_path)
    image2caption = evaluate(model, image_path, vocab, device)
    print(image2caption)

    









    
