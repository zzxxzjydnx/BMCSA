# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:27:48 2020

@author: zhaog
"""
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import LCQMC_Dataset, load_embeddings
from utils import train, validate
from model import ABCNN

def main(train_file, dev_file, embeddings_file, vocab_file, target_dir, 
         max_length=50,
         epochs=20,
         batch_size=128,
         lr=0.0005,
         patience=20,
         max_grad_norm=10.0,
         gpu_index=0,
         checkpoint=None):
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = LCQMC_Dataset(train_file, vocab_file, max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = LCQMC_Dataset(dev_file, vocab_file, max_length)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    embeddings = load_embeddings(embeddings_file)
    model = ABCNN(embeddings, device=device).to(device)
    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
     # Compute loss and accuracy before starting (or resuming) training.
    epoch_accuracy,report,precision_scores,recall_scores,f1_scores,epoch_loss= validate(model, dev_loader, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%".format(epoch_loss, epoch_accuracy)) 
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ABCNN model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       criterion, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_accuracy,report,precision_scores,recall_scores,f1_scores,epoch_loss= validate(model, dev_loader, criterion)
        valid_losses.append(epoch_loss)
        print('=*'*50)
        print(f'epoch:{epoch}, valid_acc{epoch_accuracy}')
        print("精确率:",precision_scores)
        print("Recall:",recall_scores)
        print("F1值:",f1_scores)
        print('=*'*50)
        print(report)
        print('=*'*50)
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy. 
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch, 
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "best.pth.tar"))
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break
    
if __name__ == "__main__":
    
    main("../data/AFQMC训练集.csv","../data/AFQMC验证集.csv",
         "../data/token_vec_300.bin", "../data/vocab.txt", "models")
    