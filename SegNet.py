import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, batch_size=64, lr=1e-2, momentum=0.5, is_cuda=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.is_cuda = is_cuda
        
        # Network Architecture
        
        # SegNet Architecture
		# Takes input of size in_chn = 3 (RGB images have 3 channels)
		# Outputs size label_chn (N # of classes)
        
        # -------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------- ENCODER ---------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        
        # ENCODING consists of 5 stages
		# Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
		# Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

		# General Max Pool 2D for ENCODING layers
		# Pooling indices are stored for Upsampling in DECODING layers
        
        self.pooling_enc = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.conv_enc_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bnorm_enc_1_1 = nn.BatchNorm2d(num_features=64, momentum=self.momentum)
        self.conv_enc_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bnorm_enc_1_2 = nn.BatchNorm2d(num_features=64, momentum=self.momentum)
        
        self.conv_enc_2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bnorm_enc_2_1 = nn.BatchNorm2d(num_features=128, momentum=self.momentum)
        self.conv_enc_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bnorm_enc_2_2 = nn.BatchNorm2d(num_features=128, momentum=self.momentum)
        
        self.conv_enc_3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bnorm_enc_3_1 = nn.BatchNorm2d(num_features=256, momentum=self.momentum)
        self.conv_enc_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bnorm_enc_3_2 = nn.BatchNorm2d(num_features=256, momentum=self.momentum)
        self.conv_enc_3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bnorm_enc_3_3 = nn.BatchNorm2d(num_features=256, momentum=self.momentum)
        
        self.conv_enc_4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_enc_4_1 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_enc_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_enc_4_2 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_enc_4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_enc_4_3 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        
        self.conv_enc_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_enc_5_1 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_enc_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_enc_5_2 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_enc_5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_enc_5_3 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        
        # -------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------- DECODER ---------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        
        # DECODING consists of 5 stages
		# Each stage corresponds to their respective counterparts in ENCODING
		# General Max Pool 2D/Upsampling for DECODING layers
        
        self.unpool_dec = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.conv_dec_5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_dec_5_3 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_dec_5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_dec_5_2 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_dec_5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_dec_5_1 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        
        self.conv_dec_4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_dec_4_3 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_dec_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bnorm_dec_4_2 = nn.BatchNorm2d(num_features=512, momentum=self.momentum)
        self.conv_dec_4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bnorm_dec_4_1 = nn.BatchNorm2d(num_features=256, momentum=self.momentum)
        
        self.conv_dec_3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bnorm_dec_3_3 = nn.BatchNorm2d(num_features=256, momentum=self.momentum)
        self.conv_dec_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bnorm_dec_3_2 = nn.BatchNorm2d(num_features=256, momentum=self.momentum)
        self.conv_dec_3_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bnorm_dec_3_1 = nn.BatchNorm2d(num_features=128, momentum=self.momentum)
        
        self.conv_dec_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bnorm_dec_2_2 = nn.BatchNorm2d(num_features=128, momentum=self.momentum)
        self.conv_dec_2_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bnorm_dec_2_1 = nn.BatchNorm2d(num_features=64, momentum=self.momentum)
        
        self.conv_dec_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bnorm_dec_1_2 = nn.BatchNorm2d(num_features=64, momentum=self.momentum)
        self.conv_dec_1_1 = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.bnorm_dec_1_1 = nn.BatchNorm2d(num_features=self.out_channels, momentum=self.momentum)
        
    def forward(self, data):
        
        # ------------------------------------------------- ENCODER ---------------------------------------------------
        data = F.relu(self.bnorm_enc_1_1(self.conv_enc_1_1(data)))
        data = F.relu(self.bnorm_enc_1_2(self.conv_enc_1_2(data)))
        data, enc_idx_1 = self.pooling_enc(data)
        enc_size_1 = data.size()
        
        data = F.relu(self.bnorm_enc_2_1(self.conv_enc_2_1(data)))
        data = F.relu(self.bnorm_enc_2_2(self.conv_enc_2_2(data)))
        data, enc_idx_2 = self.pooling_enc(data)
        enc_size_2 = data.size()
        
        data = F.relu(self.bnorm_enc_3_1(self.conv_enc_3_1(data)))
        data = F.relu(self.bnorm_enc_3_2(self.conv_enc_3_2(data)))
        data = F.relu(self.bnorm_enc_3_3(self.conv_enc_3_3(data)))
        data, enc_idx_3 = self.pooling_enc(data)
        enc_size_3 = data.size()
        
        data = F.relu(self.bnorm_enc_4_1(self.conv_enc_4_1(data)))
        data = F.relu(self.bnorm_enc_4_2(self.conv_enc_4_2(data)))
        data = F.relu(self.bnorm_enc_4_3(self.conv_enc_4_3(data)))
        data, enc_idx_4 = self.pooling_enc(data)
        enc_size_4 = data.size()
        
        data = F.relu(self.bnorm_enc_5_1(self.conv_enc_5_1(data)))
        data = F.relu(self.bnorm_enc_5_2(self.conv_enc_5_2(data)))
        data = F.relu(self.bnorm_enc_5_3(self.conv_enc_5_3(data)))
        data, enc_idx_5 = self.pooling_enc(data)
        enc_size_5 = data.size()
        
        # ------------------------------------------------- DECODER ---------------------------------------------------
        
        data = self.unpool_dec(data, enc_idx_5, output_size=enc_size_4)
        data = F.relu(self.bnorm_dec_5_3(self.conv_dec_5_3(data)))
        data = F.relu(self.bnorm_dec_5_2(self.conv_dec_5_2(data)))
        data = F.relu(self.bnorm_dec_5_1(self.conv_dec_5_1(data)))
        
        data = self.unpool_dec(data, enc_idx_4, output_size=enc_size_3)
        data = F.relu(self.bnorm_dec_4_3(self.conv_dec_4_3(data)))
        data = F.relu(self.bnorm_dec_4_2(self.conv_dec_4_2(data)))
        data = F.relu(self.bnorm_dec_4_1(self.conv_dec_4_1(data)))
        
        data = self.unpool_dec(data, enc_idx_3, output_size=enc_size_2)
        data = F.relu(self.bnorm_dec_3_3(self.conv_dec_3_3(data)))
        data = F.relu(self.bnorm_dec_3_2(self.conv_dec_3_2(data)))
        data = F.relu(self.bnorm_dec_3_1(self.conv_dec_3_1(data)))
        
        data = self.unpool_dec(data, enc_idx_2, output_size=enc_size_1)
        data = F.relu(self.bnorm_dec_2_2(self.conv_dec_2_2(data)))
        data = F.relu(self.bnorm_dec_2_1(self.conv_dec_2_1(data)))
        
        data = self.unpool_dec(data, enc_idx_1)
        data = F.relu(self.bnorm_dec_1_2(self.conv_dec_1_2(data)))
        data = self.conv_dec_1_1(data)
        
        return data
