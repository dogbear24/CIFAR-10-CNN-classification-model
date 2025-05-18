# Import dependencies
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.transforms import ToTensor
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from timeit import default_timer as Time
from tqdm.auto import tqdm
from pathlib import Path
from torcheval.metrics import TopKMultilabelAccuracy



cifar10_train_dataset = datasets.CIFAR10(root='./data',
                                 train=True,
                                 download=True,
                                 transform=ToTensor())

cifar10_test_dataset = datasets.CIFAR10(root='./data',
                                 train=False,
                                 download=True,
                                 transform=ToTensor())

loaded_data_train = DataLoader(cifar10_train_dataset,
                                          batch_size=10,
                                          num_workers=2,
                                          shuffle=True)

loaded_data_test = DataLoader(cifar10_test_dataset,
                                          batch_size=10,
                                          num_workers=2,
                                          shuffle=True)

# image_tensor, label = cifar10_train_dataset[0]
# print(cifar10_train_dataset.classes[label])
# plt.imshow(image_tensor.permute(1, 2, 0))
# plt.show()
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


class CIFAR10CNNModedl0(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layers(x)
    


#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


model0 = CIFAR10CNNModedl0(3072, 6144, 10).to(device=torch.device(device))





loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(params=model0.parameters(),
                            lr=0.01)

def main():
    #metric = TopKMultilabelAccuracy(k = len(loaded_data_train[0]))

    #Training loop
    epochs = 4

    start_time = Time()
    for epoch in tqdm(range(epochs)):

        #Get loss from each batch and add it all up
        total_train_loss, total_train_acc = 0, 0

        for batch, (x, y) in enumerate(loaded_data_train):

            x, y = x.to(torch.device(device)), y.to(torch.device(device))

            model0.train()


            #forward pass
            y_pred = model0(x)

            #calculates loss
            loss = loss_fn(y_pred, y)
            total_train_loss += loss
            total_train_acc += accuracy_fn(y_true=y, y_pred=y_pred)

            #optimizer zero grad
            optimizer.zero_grad()

            #loss backward
            loss.backward()

            #optimizer step

            optimizer.step()
        #finds average of loss
        total_train_loss /= len(loaded_data_train)
        total_train_acc /= len(loaded_data_train)



        total_test_loss, total_test_accuracy = 0, 0

        #Testing
        model0.eval()

        with torch.inference_mode():
            for x, y in loaded_data_test:
                x, y = x.to(torch.device(device)), y.to(torch.device(device))

                test_pred = model0(x)

                total_test_loss += loss_fn(test_pred, y)
                total_test_accuracy += accuracy_fn(y_true=y, y_pred=test_pred)

            total_test_loss /= len(loaded_data_test)
            total_test_accuracy /= len(loaded_data_test)

        print(f"Train Loss: {total_train_loss}, Test Loss: {total_test_loss}")
        print(f"Test Acc: {total_train_acc}, Test Acc:{total_test_accuracy}")

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







