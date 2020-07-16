import torch
import copy
import time


# modified trainer from https://github.com/usuyama/pytorch-unet
def train_model(model, optimizer, scheduler, num_epochs=25, save_best=True):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    epoch_loss_list_train = []
    epoch_loss_list_val = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        optimizer.step()
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = [0, 0, 0]
            epoch_samples = 0

            for clean_data, noised_data in dataloaders[phase]:
                clean_data, noised_data = clean_data.to(device), noised_data.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(noised_data)
                    
                    loss, loss1, loss2 = wSDRloss(noised_data, clean_data, pred)
                    loss_item = loss.data.cpu().numpy()
                    loss1_item = loss1.data.cpu().numpy()
                    loss2_item = loss2.data.cpu().numpy()

                    metrics[0] += loss_item * pred.shape[0]
                    metrics[1] += loss1_item * pred.shape[0]
                    metrics[2] += loss2_item * pred.shape[0]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += pred.shape[0]

            epoch_loss = metrics[0] / epoch_samples
            epoch_loss1 = metrics[1] / epoch_samples
            epoch_loss2 = metrics[2] / epoch_samples

            if phase == 'train':
                epoch_loss_list_train.append(epoch_loss)
            else:
                epoch_loss_list_val.append(epoch_loss)

            print(phase+'_epoch_loss='+str(epoch_loss), str(epoch_loss1), str(epoch_loss2))
            # deep copy the model
            if save_best and phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    if save_best:
        model.load_state_dict(best_model_wts)
    
    print('TRAIN LOSS')
    plt.plot(epoch_loss_list_train)
    plt.pause(0.001)
    
    print('VAL LOSS')
    plt.plot(epoch_loss_list_val)
    plt.pause(0.001)
    
    return model
