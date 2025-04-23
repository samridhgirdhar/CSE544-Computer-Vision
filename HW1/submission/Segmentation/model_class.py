import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from torchvision import models,transforms
class SegNet_Encoder(nn.Module):

    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Encoder, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)
    def forward(self,x):
        
        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()
        
        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        
        return x,[ind1,ind2,ind3,ind4,ind5],[size1,size2,size3,size4,size5]


class SegNet_Decoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Decoder, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        # Stage 5
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)

        # Stage 4
        self.ConvDe41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe43 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(256, momentum=BN_momentum)

        # Stage 3
        self.ConvDe31 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe33 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(128, momentum=BN_momentum)

        # Stage 2
        self.ConvDe21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe22 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(64, momentum=BN_momentum)

        # Stage 1
        self.ConvDe11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe12 = nn.Conv2d(64, out_chn, kernel_size=3, padding=1)

    def forward(self, x, indices, sizes):
        ind1, ind2, ind3, ind4, ind5 = indices

        # Stage 5 decoding
        x = self.MaxDe(x, ind5)
        x = F.relu(self.BNDe51(self.ConvDe51(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe53(self.ConvDe53(x)))

        # Handle potential dimension mismatch before stage 4
        target_size = ind4.size()
        if x.size() != target_size:
            x = F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

        # Stage 4 decoding
        x = self.MaxDe(x, ind4)
        x = F.relu(self.BNDe41(self.ConvDe41(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe43(self.ConvDe43(x)))

        # Handle potential dimension mismatch before stage 3
        target_size = ind3.size()
        if x.size() != target_size:
            x = F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

        # Stage 3 decoding
        x = self.MaxDe(x, ind3)
        x = F.relu(self.BNDe31(self.ConvDe31(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe33(self.ConvDe33(x)))

        # Handle potential dimension mismatch before stage 2
        target_size = ind2.size()
        if x.size() != target_size:
            x = F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

        # Stage 2 decoding
        x = self.MaxDe(x, ind2)
        x = F.relu(self.BNDe21(self.ConvDe21(x)))
        x = F.relu(self.BNDe22(self.ConvDe22(x)))

        # Handle potential dimension mismatch before stage 1
        target_size = ind1.size()
        if x.size() != target_size:
            x = F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

        # Stage 1 decoding
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe11(self.ConvDe11(x)))
        x = self.ConvDe12(x)

        return x


class SegNet_Pretrained(nn.Module):
    def __init__(self,encoder_weight_pth,in_chn=3, out_chn=32):
        super(SegNet_Pretrained, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.encoder=SegNet_Encoder(in_chn=self.in_chn,out_chn=self.out_chn)
        self.decoder=SegNet_Decoder(in_chn=self.in_chn,out_chn=self.out_chn)
        encoder_state_dict = torch.load(encoder_weight_pth,weights_only=True)

        # Load weights into the encoder
        self.encoder.load_state_dict(encoder_state_dict)

        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self,x):
        x,indexes,sizes=self.encoder(x)
        x=self.decoder(x,indexes,sizes)
        return x



class DeepLabV3:
    def __init__(self, num_classes=32):
        # Initialize wandb
        wandb.init(project="deeplabv3-camvid", name="fine-tuning")
        
        # Download and load pre-trained DeepLabV3
        print("Downloading pre-trained DeepLabV3 model...")
        self.model = deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        print("Model downloaded successfully!")
        
        # Modify the classifier for CamVID classes
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def train_step(self, images, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(images)['out']
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_loader, num_epochs=10):
        print("Starting training...")
        for epoch in range(num_epochs):
            train_loss = 0.0
            num_batches = len(train_loader)
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    batch_loss = self.train_step(images, labels)
                    train_loss += batch_loss
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': batch_loss})
            
            # Calculate average loss for the epoch
            avg_train_loss = train_loss / num_batches
            
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
            })
            
            print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}')
            
            # Save model checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_train_loss,
                }, f'checkpoint_epoch_{epoch+1}.pth')
                
    def evaluate_testset(self, test_loader, num_classes=32):
        """
        Gathers predictions on the test set, computes pixel-wise metrics:
          - Pixel Accuracy
          - Classwise IoU & mIoU
          - Classwise Dice
          - Classwise Precision, Recall
          - Binning IoUs in [0,1] with 0.1 intervals
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []

        # 1) Gather predictions & ground truth
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)['out']
                preds = torch.argmax(outputs, dim=1)  # [B,H,W]
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)   # shape [N, H, W]
        all_labels = np.array(all_labels) # shape [N, H, W]

        
        flat_preds = all_preds.reshape(-1)
        flat_labels = all_labels.reshape(-1)

        
        conf_mat = confusion_matrix(flat_labels, flat_preds, labels=range(num_classes))

        
        pixel_acc = conf_mat.trace() / conf_mat.sum()

        # Classwise stats
        class_iou = np.zeros(num_classes)
        class_dice = np.zeros(num_classes)
        class_prec = np.zeros(num_classes)
        class_recall = np.zeros(num_classes)

        for c in range(num_classes):
            TP = conf_mat[c, c]
            FP = conf_mat[:, c].sum() - TP
            FN = conf_mat[c, :].sum() - TP
            

            denom_iou = (TP + FP + FN)
            class_iou[c] = TP / denom_iou if denom_iou>0 else np.nan

            denom_dice = (2*TP + FP + FN)
            class_dice[c] = (2*TP)/denom_dice if denom_dice>0 else np.nan

            denom_prec = (TP + FP)
            class_prec[c] = TP/denom_prec if denom_prec>0 else np.nan

            denom_rec = (TP + FN)
            class_recall[c] = TP/denom_rec if denom_rec>0 else np.nan

        mIoU = np.nanmean(class_iou)

        print("=== DeepLabV3 Test Set Performance ===")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        print(f"mIoU: {mIoU:.4f}\n")

        for c in range(num_classes):
            print(f"Class {c}: "
                  f"IoU={class_iou[c]:.3f}, "
                  f"Dice={class_dice[c]:.3f}, "
                  f"Prec={class_prec[c]:.3f}, "
                  f"Recall={class_recall[c]:.3f}")
        print("=======================================")

        # 4) Bin IoUs in [0,1] with 0.1 intervals
        bin_edges = np.linspace(0,1,11)  # 0.0..0.1..1.0
        hist, _ = np.histogram(class_iou, bins=bin_edges)
        for i in range(len(hist)):
            print(f"Classes with IoU in [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {hist[i]}")
        
        
        return {
            'pixel_acc': pixel_acc,
            'class_iou': class_iou,
            'mIoU': mIoU,
            'class_dice': class_dice,
            'class_prec': class_prec,
            'class_recall': class_recall
        }