import os
import spacy
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence  # pad batch
import pandas as pd


spacy_eng = spacy.load("en")


class Vocabulary:

    def __init__(self, freq_threshold):
        
        self.freq_threshold = freq_threshold

        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0,"<SOS>":1, "<EOS>":2, "<UNK>":3}
        
    
    def __len__(self):
        return len(self.itos)
    

    def tokenizer(self, text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                    
                else:
                    frequencies[word] += 1
                    
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                 
                    
    def numericalize(self, text):
        
        tokenized_text = self.tokenizer(text)
        
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]
    
    


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        super().__init__()
        
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.transform = transform
        self.freq_threshold = freq_threshold
        
        self.df = pd.read_csv(captions_file)
        
        self.images = self.df['image']
        self.captions = self.df['caption']
        
        self.vocab = Vocabulary(freq_threshold=freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

        torch.save(self.vocab, 'vocab.pt')
        
    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        
        caption = self.captions[index]
        image_id = self.images[index]
        
        image = Image.open(os.path.join(self.root_dir, image_id)).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        numericalized_caption = [self.vocab.stoi['<SOS>']]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<EOS>'])

        return image, torch.tensor(numericalized_caption)
    

class collate_fn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        
        return imgs, targets
        
        

def get_dataloader(root_dir, annotation_file, transform, batch_size, shuffle=True, pin_memory=True):
    
    dataset = FlickerDataset(root_dir, annotation_file, transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn(pad_idx=pad_idx))
    
    return loader, dataset
    
        
        
        
def save_checkpoint(state, filename="my_checkpoint.pt"):
    print("=> Saving checkpoint")
    torch.save(state, filename)        

        
        
        
    
    