import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split  
from torchvision import transforms  
import os
from PIL import Image
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, nc=1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                        

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                      

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                           

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          

            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          

            nn.AdaptiveAvgPool2d((1, 1))                
        )

    def forward(self, x):
        return self.encoder(x) 

class Decoder(nn.Module):
    
    def __init__(self, nc=1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(True),

            nn.Conv2d(16, nc, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    
    def __init__(self, nc=1):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(nc)
        self.decoder = Decoder(nc)

    def forward(self, x):
        encoded = self.encoder(x)        
        reconstructed = self.decoder(encoded)
        return reconstructed  

class AutoencoderFeat(nn.Module):
    
    def __init__(self, nc=1, output_dim=8, n_morph=6):
        super(AutoencoderFeat, self).__init__()
        self.encoder = Encoder(nc)
        self.decoder = Decoder(nc)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, output_dim)
        )
        self.feature_head = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
            nn.Linear(output_dim, n_morph)
        ) 

    def forward(self, x):
        encoded = self.encoder(x)         
        reconstructed = self.decoder(encoded)
        z = self.fc(encoded.squeeze(-1).squeeze(-1))
        feat = self.feature_head(z)
        return reconstructed, feat, z 

class AutoencoderLabel(nn.Module):

    def __init__(self, nc=1, output_dim=128, n_class=7):
        super().__init__()
        self.encoder = Encoder(nc)
        self.decoder = Decoder(nc)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, output_dim),
            nn.LeakyReLU(True)
        )
        self.feature_head = nn.Sequential(
            nn.Linear(output_dim, n_class)
        ) 

    def forward(self, x):
        encoded = self.encoder(x)         
        reconstructed = self.decoder(encoded)
        z = self.fc(encoded.squeeze(-1).squeeze(-1))
        logits = self.feature_head(z)
        
        return reconstructed, logits, z  

class FeatureHead(nn.Module):
    
    def __init__(self, input_dim=2048, output_dim=128, n_morph=6):
        super(FeatureHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        self.morph_head1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 32),            
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, output_dim),
        )
        self.morph_head2 = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
            nn.Linear(output_dim, n_morph),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, output_dim),
        )                  


    def forward(self, x):
        x = self.fc(x)
        inter = self.morph_head1(x)
        morph = self.morph_head2(inter)
        feat = self.feature_head(x)
        return feat, morph, inter

class FeatureExtractor(nn.Module):
    
    def __init__(self, nc=1, input_dim=2048, feature_dim=8, n_morph=8):
        super(FeatureExtractor, self).__init__()
        self.encoder = Encoder(nc)  
        self.head = FeatureHead(input_dim=input_dim, output_dim=feature_dim, n_morph=n_morph)

    def forward(self, x):
        h = self.encoder(x)                  
        h_flat = h.view(h.size(0), -1)      
        feat, morph, head = self.head(h_flat)     
        return feat, morph, {'feat':head, 'h': h_flat}
    
