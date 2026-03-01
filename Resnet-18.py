from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''

    # Set model to train mode before each epoch
    model.train()

    losses = []
    correct = 0

    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        losses.append(loss.item())
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss = float(np.mean(losses))
    train_acc = correct / len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * train_acc))
    return train_loss, train_acc


def evaluate(model, device, loader, criterion, split_name='Test'):
    '''
    Evaluates the model on a given data loader (validation or test).
    model: The model to evaluate. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    loader: dataloader for the split to evaluate.
    criterion: loss function.
    split_name: label for print output ('Val' or 'Test').
    '''

    model.eval()

    losses = []
    correct = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(loader.dataset)

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        split_name, eval_loss, correct, len(loader.dataset), accuracy))

    return eval_loss, accuracy


def save_misclassified(model, device, test_loader, test_dataset, class_names,
                       output_dir='misclassified'):
    '''
    Runs inference on the test set, copies each misclassified image to
    output_dir/true_<TRUE>_pred_<PRED>/, and prints per-class accuracy.
    class_names: list of class name strings, e.g. ['NORMAL', 'PNEUMONIA'].
    output_dir: root folder where misclassified images are saved.
    '''
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    global_idx = 0  # tracks position across batches in the flat dataset

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            preds = model(data).argmax(dim=1)

            for i in range(len(target)):
                true_label = target[i].item()
                pred_label = preds[i].item()

                class_total[true_label] += 1
                if pred_label == true_label:
                    class_correct[true_label] += 1
                else:
                    img_path, _ = test_dataset.imgs[global_idx]
                    true_name   = class_names[true_label]
                    pred_name   = class_names[pred_label]

                    save_subdir = os.path.join(output_dir,
                                               f'true_{true_name}_pred_{pred_name}')
                    os.makedirs(save_subdir, exist_ok=True)
                    shutil.copy(img_path,
                                os.path.join(save_subdir, os.path.basename(img_path)))

                global_idx += 1

    print("\n--- Per-Class Accuracy (Test Set) ---")
    for c in range(num_classes):
        acc = 100. * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        print(f"  {class_names[c]:<12}: {class_correct[c]}/{class_total[c]} ({acc:.1f}%)")

    total_correct = sum(class_correct)
    total         = sum(class_total)
    print(f"  {'Overall':<12}: {total_correct}/{total} "
          f"({100. * total_correct / total:.1f}%)")
    print(f"\n{total - total_correct} misclassified images saved to '{output_dir}/'")


def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)

    # -----------------------------------------------------------------------
    #   Build ResNet-18 
    #   mode=1  Task 1.1 – train from scratch (random initialisation)
    #   mode=2  Task 1.2 – fine-tune from ImageNet pretrained weights
    # -----------------------------------------------------------------------
    if FLAGS.mode == 1:
        mode_label = 'Task1.1 - From Scratch'
        model = models.resnet18(weights=None)          # random initialisation
        print("Mode 1: Training ResNet-18 from scratch (random weights)")
    elif FLAGS.mode == 2:
        mode_label = 'Task1.2 - Fine-tuned'
        model = models.resnet18(                       # ImageNet pretrained
            weights=models.ResNet18_Weights.DEFAULT)
        print("Mode 2: Fine-tuning pretrained ResNet-18 (ImageNet weights)")
    else:
        print(f"Invalid mode {FLAGS.mode}. Choose 1 (scratch) or 2 (fine-tune).")
        return

    num_classes = 2  # NORMAL vs PNEUMONIA Classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=FLAGS.learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4)

    # -----------------------------------------------------------------------
    # Transforms: ResNet-18 expects 224x224 RGB images with ImageNet stats
    # -----------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------------------------------------------------
    # Load data from train/, val/, and test/ folders using ImageFolder.
    # Each subfolder name becomes a class label (NORMAL=0, PNEUMONIA=1).
    # -----------------------------------------------------------------------
    train_dataset = datasets.ImageFolder(root='train', transform=train_transform)
    val_dataset   = datasets.ImageFolder(root='val',   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root='test',  transform=eval_transform)

    print("Classes:", train_dataset.classes)
    print(f"Train samples: {len(train_dataset)}, "
          f"Val samples: {len(val_dataset)}, "
          f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=FLAGS.batch_size,
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=FLAGS.batch_size,
                              shuffle=False, num_workers=4)

    best_accuracy = 0.0

    train_losses, val_losses,  test_losses = [], [], []
    train_accs,   val_accs,    test_accs   = [], [], []

    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{FLAGS.num_epochs} ---")
        train_loss, train_acc = train(model, device, train_loader,
                                      optimizer, criterion, epoch, FLAGS.batch_size)
        val_loss,   val_acc   = evaluate(model, device, val_loader,   criterion, 'Val')
        test_loss,  test_acc  = evaluate(model, device, test_loader,  criterion, 'Test')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        train_accs.append(train_acc * 100.0)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if test_acc > best_accuracy:
            best_accuracy = test_acc

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses,   label='Val Loss',   linewidth=2)
    axes[0].plot(epochs, test_losses,  label='Test Loss',  linewidth=2)
    axes[0].set_title(f'ResNet-18 ({mode_label}) - Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend(loc='best')

    axes[1].plot(epochs, train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs,   label='Val Acc',   linewidth=2)
    axes[1].plot(epochs, test_accs,  label='Test Acc',  linewidth=2)
    axes[1].set_title(f'ResNet-18 ({mode_label}) - Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True)
    axes[1].legend(loc='best')

    plt.tight_layout()
    out_path = f'resnet18_mode{FLAGS.mode}_curves.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print("Best test accuracy: {:2.2f}%".format(best_accuracy))

    save_misclassified(model, device, test_loader, test_dataset,
                       class_names=test_dataset.classes,
                       output_dir=f'misclassified_mode{FLAGS.mode}')

    print("Training and evaluation finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ResNet-18 Pneumonia Classification.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='1 = train from scratch, 2 = fine-tune pretrained ImageNet weights.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int, default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--log_dir',
                        type=str, default='logs',
                        help='Directory to put logging.')

    FLAGS, unparsed = parser.parse_known_args()
    run_main(FLAGS)
