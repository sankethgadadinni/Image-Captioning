import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models




class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super(EncoderCNN, self).__init__()
        
        self.embedding_size = embedding_size
        
        self.inception = models.inception_v3(pretrained=True, aux_logits = False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_size)    #in_features = 2048
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self, images):
        
        # images shape :: [batch size, channels, height, width]
        features = self.inception(images)
        #features shape :: [batch size, embedding size]
        features = self.dropout(self.relu(features))
        
        return features
    
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    
    def forward(self, features, captions):
        
        # features shape :: [batch size, embedding size]
        embeddings = self.dropout(self.embedding(captions))

        # embeeding shape :: [seq_length, batch size, embeeding size]
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        #h hiddens shape :: [seq length, batch size, hidden size]
        hiddens, _ = self.lstm(embeddings)

        # outputs shape :: [seq length, batch size, vocab size]
        outputs = self.linear(hiddens)
        
        return outputs
        


class CNNtoRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.encoder = EncoderCNN(embedding_size=embedding_size)
        self.decoder = DecoderRNN(embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers)
        
    
    def forward(self, images, captions):
        
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        
        return outputs
    

    def caption_image(self, image, vocab, device, max_length=50):
        result_caption = []
        
        with torch.no_grad():
            image = image.unsqueeze(0)
            image = image.to(device)
            x = self.encoder(image)
            x = x.unsqueeze(0)

            states = None
                        
            for i in range(max_length):
                
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                
                x = self.decoder.embedding(predicted).unsqueeze(0)
                
                if vocab.itos[predicted.item()] == '<EOS>':
                    break
                
            caption = [vocab.itos[x] for x in result_caption]
            caption = ' '.join([id for id in caption])

            caption = caption[6:-6]

        return caption
                
        
        
        
        