import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network with exactly 4 hidden layers using 
        a smoother progression toward the output classes.

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        # MLP with 4 hidden layers - smoother transition to output
        self.fc1 = nn.Linear(input_size, 512)      # First hidden layer (wider)
        self.bn1 = nn.BatchNorm1d(512)             # Batch normalization for stability
        self.fc2 = nn.Linear(512, 256)             # Second hidden layer (reducing)
        self.bn2 = nn.BatchNorm1d(256)             # Batch normalization
        self.fc3 = nn.Linear(256, 128)             # Third hidden layer (reducing further)
        self.bn3 = nn.BatchNorm1d(128)             # Batch normalization
        self.fc4 = nn.Linear(128, 64)              # Fourth hidden layer (closer to n_classes)
        self.bn4 = nn.BatchNorm1d(64)              # Batch normalization
        self.fc5 = nn.Linear(64, n_classes)        # Output layer
        self.relu = nn.ReLU()                      # ReLU activation for hidden layers
        self.dropout = nn.Dropout(0.3)             # Dropout for regularization

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # Forward pass through all layers with batch normalization
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        preds = self.fc5(x)  # Final output logits
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        # Block 1
        self.conv1a = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(16)
        
        # Block 2
        self.conv2a = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(32)
        
        # Block 3
        self.conv3a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(64)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Global Average Pooling to (1, 1) spatial size
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)
        x = self.dropout(x)

        # Global average pooling
        x = self.gap(x)             
        x = x.view(x.size(0), -1)

        # Final fully connected layer
        x = self.fc(x)
        return x
class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, device):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model.to(device)
        self.device  = device
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        #sochastic gradient descent optimization that we are using rn
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.train_losses = []

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            avg_loss = self.train_one_epoch(dataloader, ep)
            self.train_losses.append(avg_loss)
            print(f"Epoch {ep+1}/{self.epochs} - Average Loss: {avg_loss:.4f}")

            #ici je vais devoir log les precisions de nos models et ptt log la time complexity ??



    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        #comme demandé on met le trainer en training mode 
        self.model.train(mode = True) ## je pense qu'apres on va devoir lire ce self et
        total_loss = 0 
        

        for batch_x, batch_y in dataloader :
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            pds = self.model(batch_x)
            loss = self.criterion(pds,batch_y)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item() 

        avg_loss = total_loss / len(dataloader)
        return avg_loss

        
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()

        pred_list = []
        with torch.no_grad():
            for batch in dataloader :
                batch_x = batch[0].to(self.device)
                output = self.model(batch_x)
                predicted = torch.argmax(output, dim=1)

                pred_list.append(predicted.cpu())

        pred_labels = torch.cat(pred_list, dim=0) 

            
        

        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        dummy_labels = torch.zeros(test_data.shape[0], dtype=torch.long) 
        test_dataset = TensorDataset(torch.from_numpy(test_data).float(), dummy_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)
        return pred_labels.cpu().numpy()
    
    def plot_training_progress(self, train_losses, val_accuracies=None):
        """
        Plot training loss (and optional validation accuracy) over epochs.

        Arguments:
            train_losses (list of float): training loss for each epoch
            val_accuracies (list of float, optional): validation accuracy per epoch
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        
        if val_accuracies is not None:
            plt.plot(val_accuracies, label='Validation Accuracy')

        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
