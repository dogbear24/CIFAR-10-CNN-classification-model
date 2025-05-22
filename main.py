# Import dependencies
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from timeit import default_timer as Time
from tqdm.auto import tqdm
from pathlib import Path
from torcheval.metrics import TopKMultilabelAccuracy
import random


#Augments the data
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010])
])




cifar10_train_dataset = datasets.CIFAR10(root='./data',
                                 train=True,
                                 download=True,
                                 transform=train_transform)

cifar10_test_dataset = datasets.CIFAR10(root='./data',
                                 train=False,
                                 download=True,
                                 transform=train_transform)

loaded_data_train = DataLoader(cifar10_train_dataset,
                                          batch_size=128,
                                          num_workers=2,
                                          shuffle=True)

loaded_data_test = DataLoader(cifar10_test_dataset,
                                          batch_size=128,
                                          num_workers=2,
                                          shuffle=True)

# image_tensor, label = cifar10_train_dataset[0]
# print(cifar10_train_dataset.classes[label])
# plt.imshow(image_tensor.permute(1, 2, 0))
# plt.show()
# print(image_tensor.shape)

def predict_and_plot_batch(model, dataset, class_names, device, batch_size=10):
    """
    Predicts and plots a batch of images from the dataset.

    Args:
        model (torch.nn.Module): Trained model for inference.
        dataset (torchvision.datasets): Dataset to pull images from.
        class_names (list): Class names for CIFAR-10.
        device (str): "cuda" or "cpu".
        batch_size (int): Number of images to show and predict.
    """
    model.eval()

    indices = random.sample(range(len(dataset)), batch_size)

    images, labels = zip(*[dataset[i] for i in indices])
    image_batch = torch.stack(images).to(device=device)
    label_batch = torch.tensor(labels).to(device)

    with torch.inference_mode():
        logits = model(image_batch)
    preds = torch.argmax(logits, dim=1)

    fig = plt.figure(figsize=(28, 5))
    for i in range(batch_size):
        ax = fig.add_subplot(1, batch_size, i + 1)
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(f"Pred: {class_names[preds[i]]}\n True: {class_names[labels[i]]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(torch.device(device)), y.to(torch.device(device))

            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}



class CIFAR10CNNModedl0(nn.Module):
    # def __init__(self, in_features, hidden_units, out_features):
    #     super().__init__()
        
    #     self.block_1 = nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(in_features=in_features, 
    #                   out_features=hidden_units),
    #         nn.ReLU(),
    #         nn.Linear(in_features=hidden_units, out_features=out_features),
    #         #nn.Softmax(dim=1)
            
    #     )
    
    # def forward(self, x: torch.Tensor):
    #     return self.block_1(x)
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units * 8 * 8, 
                      out_features=out_features)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
    


#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


model0 = CIFAR10CNNModedl0(3, 64, 10).to(device=torch.device(device))





loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model0.parameters(),
                            lr=0.01)

def main():
    #metric = TopKMultilabelAccuracy(k = len(loaded_data_train[0]))

    #Training loop
    epochs = 50

    start_time = Time()
    for epoch in tqdm(range(epochs)):

        #Get loss from each batch and add it all up
        total_train_loss = 0

        for batch, (x, y) in enumerate(loaded_data_train):

            x, y = x.to(torch.device(device)), y.to(torch.device(device))

            model0.train()

    
            #forward pass
            y_pred = model0(x)

            #calculates loss
            loss = loss_fn(y_pred, y)
            total_train_loss += loss

            #optimizer zero grad
            optimizer.zero_grad()

            #loss backward
            loss.backward()

            #optimizer step

            optimizer.step()
        #finds average of loss
        total_train_loss /= len(loaded_data_train)



        total_test_loss = 0

        #Testing
        model0.eval()

        with torch.inference_mode():
            for x, y in loaded_data_test:
                x, y = x.to(torch.device(device)), y.to(torch.device(device))

                test_pred = model0(x)

                total_test_loss += loss_fn(test_pred, y)

            total_test_loss /= len(loaded_data_test)

        print(f"Epoch: {epoch}, Time: {Time() - start_time}, Train Loss: {total_train_loss}, Test Loss: {total_test_loss}")
        # Calculate model 0 results on test dataset
        
        model_0_results = eval_model(model=model0, data_loader=loaded_data_test,
        loss_fn=loss_fn, accuracy_fn=accuracy_fn)
        print(model_0_results)
                

    end_time = Time()
    print(f"Total Time:{end_time - start_time}")

    #metric.compute()

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    
    path = Path("Models")
    if not path.exists():
        path.mkdir(exist_ok=True)


    main()

    


    torch.save(model0.state_dict(), 'Models/model0.pt')

    predict_and_plot_batch(model0, cifar10_test_dataset, cifar10_test_dataset.classes, device)



    # loadedModel0 = CIFAR10CNNModedl0(3072, 6144, 10)
    # loadedModel0.load_state_dict(torch.load(f="Models/model0.pt"))


    # # Calculate model 0 results on test dataset
    # model_0_results = eval_model(model=loadedModel0, data_loader=loaded_data_test,
    #     loss_fn=loss_fn, accuracy_fn=accuracy_fn
    # )
    # print(model_0_results)



