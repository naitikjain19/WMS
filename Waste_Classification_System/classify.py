import torch
import torch.nn as nn
import os
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn.functional as F
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from flask import Blueprint, flash
from .models import User
from . import db
from flask_login import login_required, current_user


transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])



# Define a flask app
# app = Flask(__name__)
classify = Blueprint('classify', __name__)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self, n_classes):
        super().__init__()
        
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
    
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, n_classes)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model = ResNet(n_classes=len(CLASSES))
model.load_state_dict(torch.load(r'C:\Users\Naitik Jain\Desktop\WMS\project\model.pth', map_location=torch.device('cpu')))
model.eval()
# print('Model loaded. Check http://127.0.0.1:5000/')

@classify.route('/classification', methods=['GET'])
def classification():
    # Main page
    return render_template('classify.html')


@classify.route('/prediction', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        return result
    return None

def model_predict(img_path, model):
    image = Image.open(img_path)
    image = transformations(image)
    return predict_image(image)


def predict_image(img):
    # Convert to a batch of 1
    xb = img.unsqueeze(0)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return CLASSES[preds[0].item()]

if __name__ == '__main__':
    classify.run(debug=True)

