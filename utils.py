import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize

RegNet_transform = Compose([
    ToTensor(),
    Resize((232,232)),
    CenterCrop((224,224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

EfficientNet_transform = Compose([
    ToTensor(),
    Resize((384,384)),
    CenterCrop((384,384)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SwinV2_transform = Compose([
    ToTensor(),
    Resize((260,260)),
    CenterCrop((256,256)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset_FolderLabels:
    """
    CustomDataset class for loading and splitting a dataset into training, validation, and testing sets.

    Args:
        data_path (str): Path to the main folder containing subfolders for each class.
        train_ratio (float): Ratio of data allocated for the training set (0.0 to 1.0).
        val_ratio (float): Ratio of data allocated for the validation set (0.0 to 1.0).
        test_ratio (float): Ratio of data allocated for the testing set (0.0 to 1.0).
        batch_size (int): Number of samples per batch in the data loaders.
        transform (torchvision.transforms.transforms.Compose): Transformations to be applied on the image

    Attributes:
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): Data loader for the testing set.

    """
    def __init__(self, data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, transform=None):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        if transform == None:
            self.transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and splits it into training, validation, and testing sets.

        """
        dataset = ImageFolder(root=self.data_path, transform=self.transform)
        num_samples = len(dataset)

        train_size = int(self.train_ratio * num_samples)
        val_size = int(self.val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def get_train_loader(self):
        """
        Get the data loader for the training set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the training set.

        """
        return self.train_loader

    def get_val_loader(self):
        """
        Get the data loader for the validation set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the validation set.

        """
        return self.val_loader

    def get_test_loader(self):
        """
        Get the data loader for the testing set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the testing set.

        """
        return self.test_loader
    
def TrainLoop(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    num_epochs:int=20,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    device:str='cpu'
):
    """
    TrainLoop is a function for training a PyTorch model using provided data loaders and settings.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters during training.
        criterion (torch.nn.Module): The loss function used for computing the training loss.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for validation data (default: None).
        num_epochs (int, optional): The number of training epochs (default: 20).
        early_stopping_rounds (int, optional): Number of epochs without improvement to trigger early stopping (default: 5).
        device (str, optional): Device to run training on ('cpu' or 'cuda'). (default: 'cpu').

    Returns:
        None

    This function trains a PyTorch model using the specified settings, including data loaders, optimization,
    and loss function. It supports optional early stopping based on validation loss. After training,
    it displays training, testing, and validation loss plots.

    Example:
        TrainLoop(
            model=my_model,
            optimizer=my_optimizer,
            criterion=my_loss_function,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            val_dataloader=val_loader,
            num_epochs=10,
            early_stopping_rounds=3,
            device='cuda'
        )
    """
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    total_train_loss = []
    total_val_loss = []
    total_test_loss = []

    best_model_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("\nEpoch {}\n----------".format(epoch))
        train_loss = 0
        for i, (batch, label) in enumerate(train_dataloader):
            batch, label = batch.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, label.float().unsqueeze(dim=1))
            train_loss += loss
            loss.backward()
            optimizer.step()
            print("Loss for batch {} = {}".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for batch, label in val_dataloader:
                batch, label = batch.to(device), label.to(device)
                outputs = model(batch)
                loss = criterion(outputs, label.float().unsqueeze(dim=1))
                validation_loss += loss

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))

        if epochs_without_improvement >= early_stopping_rounds:
            print("Early Stoppping Triggered")
            break

    if return_best_model == True:
        model.load_state_dict(best_model_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)

    x_train = np.arange(len(total_train_loss))
    x_val = np.arange(len(total_val_loss))

    sns.lineplot(x=x_train, y=total_train_loss, label='Training Loss')
    sns.lineplot(x=x_val, y=total_val_loss, label='Validation Loss')
    plt.title("Loss over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(len(total_train_loss)))
    plt.show()


def TrainLoopv2(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    num_epochs:int=20,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    device:str='cpu'
):
    """
    Train a PyTorch model using the provided data loaders and monitor training progress.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (torch.nn.Module): The loss function to optimize.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the test dataset (default: None).
        num_epochs (int, optional): Number of training epochs (default: 20).
        early_stopping_rounds (int, optional): Number of epochs to wait for improvement in validation loss
            before early stopping (default: 5).
        return_best_model (bool, optional): Whether to return the model with the best validation loss (default: True).
        device (str, optional): Device to use for training ('cpu' or 'cuda') (default: 'cpu').

    Returns:
        None or torch.nn.Module: If return_best_model is True, returns the trained model with the best validation loss;
        otherwise, returns None.

    Note:
        This function monitors training and validation loss and accuracy over epochs and can optionally
        plot the loss and accuracy curves.

    Example:
        See the code example provided for usage.
    """
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    total_train_loss = []
    total_val_loss = []
    total_test_loss = []
    best_model_weights = model.state_dict()

    train_accuracies = []
    val_accuracies = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("\nEpoch {}\n----------".format(epoch))
        train_loss = 0
        for i, (batch, label) in enumerate(train_dataloader):
            batch, label = batch.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, label.float().unsqueeze(dim=1))
            train_loss += loss
            loss.backward()
            optimizer.step()
            print("Loss for batch {} = {}".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            val_true_labels = []
            train_true_labels = []
            val_pred_labels = []
            train_pred_labels = []
            for batch, label in val_dataloader:
                batch, label = batch.to(device), label.to(device)
                outputs = model(batch)
                loss = criterion(outputs, label.float().unsqueeze(dim=1))
                validation_loss += loss

                outputs = torch.round(torch.sigmoid(outputs))
                val_true_labels.extend(label.cpu().numpy())
                val_pred_labels.extend(outputs.cpu().numpy())

            for batch, label in train_dataloader:
                batch, label = batch.to(device), label.to(device)
                outputs = model(batch)

                outputs = torch.round(torch.sigmoid(outputs))
                train_true_labels.extend(label.cpu().numpy())
                train_pred_labels.extend(outputs.cpu().numpy())

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            val_true_labels = np.array(val_true_labels)
            train_true_labels = np.array(train_true_labels)
            val_pred_labels = np.array(val_pred_labels)
            train_pred_labels = np.array(train_pred_labels)

            train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
            val_accuracy = accuracy_score(val_true_labels, val_pred_labels)

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")

            print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))

        if epochs_without_improvement >= early_stopping_rounds:
            print("Early Stoppping Triggered")
            break

    if return_best_model == True:
        model.load_state_dict(best_model_weights)
    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)

    train_accuracies = np.array(train_accuracies)
    val_accuracies = np.array(val_accuracies)

    x_train = np.arange(len(total_train_loss))
    x_val = np.arange(len(total_val_loss))
    
    sns.set_style('whitegrid')
    plt.figure(figsize=(14,5))
    
    plt.subplot(1,2,1)
    sns.lineplot(x=x_train, y=total_train_loss, label='Training Loss')
    sns.lineplot(x=x_val, y=total_val_loss, label='Validation Loss')
    plt.title("Loss over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(len(total_train_loss)))
    
    plt.subplot(1,2,2)
    sns.lineplot(x=x_train, y=train_accuracies, label='Training Accuracy')
    sns.lineplot(x=x_val, y=val_accuracies, label='Validation Accuracy')
    plt.title("Accuracy over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(len(total_train_loss)))

    plt.show()

def TrainLoopv1_1(
    model,
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.Module,
    train_dataloader:torch.utils.data.DataLoader,
    val_dataloader:torch.utils.data.DataLoader,
    num_epochs:int=20,
    early_stopping_rounds:int=5,
    return_best_model:bool=True,
    scheduler=None,
    device:str='cpu'
):
    """
    TrainLoop is a function for training a PyTorch model using provided data loaders and settings.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters during training.
        criterion (torch.nn.Module): The loss function used for computing the training loss.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for validation data (default: None).
        num_epochs (int, optional): The number of training epochs (default: 20).
        early_stopping_rounds (int, optional): Number of epochs without improvement to trigger early stopping (default: 5).
        device (str, optional): Device to run training on ('cpu' or 'cuda'). (default: 'cpu').

    Returns:
        None

    This function trains a PyTorch model using the specified settings, including data loaders, optimization,
    and loss function. It supports optional early stopping based on validation loss. After training,
    it displays training, testing, and validation loss plots.

    Example:
        TrainLoop(
            model=my_model,
            optimizer=my_optimizer,
            criterion=my_loss_function,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            val_dataloader=val_loader,
            num_epochs=10,
            early_stopping_rounds=3,
            device='cuda'
        )
    """
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    total_train_loss = []
    total_val_loss = []
    total_test_loss = []

    best_model_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("\nEpoch {}\n----------".format(epoch))
        train_loss = 0
        for i, (batch, label) in enumerate(train_dataloader):
            batch, label = batch.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, label.float().unsqueeze(dim=1))
            train_loss += loss
            loss.backward()
            optimizer.step()
            print("Loss for batch {} = {}".format(i, loss))

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for batch, label in val_dataloader:
                batch, label = batch.to(device), label.to(device)
                outputs = model(batch)
                loss = criterion(outputs, label.float().unsqueeze(dim=1))
                validation_loss += loss

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
            best_model_weights = model.state_dict()
        else:
            epochs_without_improvement += 1

        print(f"Current Validation Loss = {validation_loss}")
        print(f"Best Validation Loss = {best_val_loss}")
        print(f"Epochs without Improvement = {epochs_without_improvement}")

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))
        
        try:
            scheduler.step(validation_loss)
        except:
            scheduler.step()
            
        if epochs_without_improvement >= early_stopping_rounds:
            print("Early Stoppping Triggered")
            break

    if return_best_model == True:
        model.load_state_dict(best_model_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)

    x_train = np.arange(len(total_train_loss))
    x_val = np.arange(len(total_val_loss))

    sns.lineplot(x=x_train, y=total_train_loss, label='Training Loss')
    sns.lineplot(x=x_val, y=total_val_loss, label='Validation Loss')
    plt.title("Loss over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(len(total_train_loss)))
    plt.show()
