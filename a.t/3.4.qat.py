import os
import time
import torch
from torchvision import transforms, datasets
import torchvision.models.quantization as models

if __name__ == '__main__':
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=8)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import torchvision

    def imshow(inp, title=None, ax=None, figsize=(5, 5)):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(inp)
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs, nrow=4)

        fig, ax = plt.subplots(1, figsize=(10, 10))
        imshow(out, title=[class_names[x] for x in classes], ax=ax)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
        """
        Support function for model training.

        Args:
            model: Model to be trained
            criterion: Optimization criterion (loss)
            optimizer: Optimizer to use for training
            scheduler: Instance of ``torch.optim.lr_scheduler``
            num_epochs: Number of epochs
            device: Device to run the training on. Must be 'cpu' or 'cuda'
        """
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def visualize_model(model, rows=3, cols=3):
        was_training = model.training
        model.eval()
        current_row = current_col = 0
        fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

        with torch.no_grad():
            for idx, (imgs, lbls) in enumerate(dataloaders['val']):
                imgs = imgs.cpu()
                lbls = lbls.cpu()

                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)

                for jdx in range(imgs.size()[0]):
                    imshow(imgs.data[jdx], ax=ax[current_row, current_col])
                    ax[current_row, current_col].axis('off')
                    ax[current_row, current_col].set_title('predicted: {}'.format(class_names[preds[jdx]]))

                    current_col += 1
                    if current_col >= cols:
                        current_row += 1
                        current_col = 0
                    if current_row >= rows:
                        model.train(mode=was_training)
                    return
                model.train(mode=was_training)

    # notice `quantize=False`
    model = models.resnet18(pretrained=True, progress=True, quantize=False)
    num_ftrs = model.fc.in_features

    # Step 1
    model.train()
    model.fuse_model()

    visualize_model(model)

    def print_size_of_model(model, label=""):
        torch.save(model.state_dict(), "temp.p")
        size=os.path.getsize("temp.p")
        print("model: ",label,' \t','Size (KB):', size/1e3)
        os.remove('temp.p')
        return size

    # compare the sizes
    f=print_size_of_model(model,"fp32")

    # compare the performance
    startTime = time.time()
    model.forward(inputs, hidden)
    executionTime = (time.time() - startTime)

    print('Floating point execution time in seconds: ' + str(executionTime))

    startTime = time.time()
    quantized_lstm.forward(inputs,hidden)
    executionTime = (time.time() - startTime)

    print('Quanitzed int execution time in seconds: ' + str(executionTime))

    out1, hidden1 = float_lstm(inputs, hidden)
    mag1 = torch.mean(abs(out1)).item()
    print('mean absolute value of output tensor values in the FP32 model is {0:.5f} '.format(mag1))

    # run the quantized model
    out2, hidden2 = quantized_lstm(inputs, hidden)
    mag2 = torch.mean(abs(out2)).item()
    print('mean absolute value of output tensor values in the INT8 model is {0:.5f}'.format(mag2))

    # compare them
    mag3 = torch.mean(abs(out1-out2)).item()
    print('mean absolute value of the difference between the output tensors is {0:.5f} or {1:.2f} percent'.format(mag3,mag3/mag1*100))

            



    # Step 2
    from torch import nn

    def create_combined_model(model_fe):
    # Step 1. Isolate the feature extractor.
        model_fe_features = nn.Sequential(
            model_fe.quant,  # Quantize the input
            model_fe.conv1,
            model_fe.bn1,
            model_fe.relu,
            model_fe.maxpool,
            model_fe.layer1,
            model_fe.layer2,
            model_fe.layer3,
            model_fe.layer4,
            model_fe.avgpool,
            model_fe.dequant,  # Dequantize the output
        )

        # Step 2. Create a new "head"
        new_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, 2),
        )

        # Step 3. Combine, and don't forget the quant stubs.
        new_model = nn.Sequential(
            model_fe_features,
            nn.Flatten(1),
            new_head,
        )
        return new_model

    model_ft = create_combined_model(model)
    model_ft[0].qconfig = torch.quantization.default_qat_qconfig  # Use default QAT configuration
    # Step 3
    model_ft = torch.quantization.prepare_qat(model_ft, inplace=True)

    for param in model_ft.parameters():
        param.requires_grad = True

    model_ft.to(device)  # We can fine-tune on GPU if available

    criterion = nn.CrossEntropyLoss()

    # Note that we are training everything, so the learning rate is lower
    # Notice the smaller learning rate
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)

    # Decay LR by a factor of 0.3 every several epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.3)

    model_ft_tuned = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=25, device=device)

    from torch.quantization import convert
    model_ft_tuned.cpu()

    model_quantized_and_trained = convert(model_ft_tuned, inplace=False)

    visualize_model(model_quantized_and_trained)

    plt.ioff()
    plt.tight_layout()
    plt.show()