import torch
import numpy as np

import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses,args):
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{args.mode}_losses.png')
    plt.close()
    

def plot_accuracies(train_accuracies, val_accuracies,args):
    plt.plot(train_accuracies, label='train')
    plt.plot(val_accuracies, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'plots/{args.mode}_accuracies.png')
    plt.close()
    


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.mps.deterministic = True


def get_data_loader(data_path, batch_size, tokenizer, shuffle=True, max_len=20):
    """
    Get a data loader for the training data.
    """
    data = np.loadtxt(data_path, delimiter='\t', dtype=str)
    X, y = data[:, -1], data[:, 1]
    X = tokenizer.batch_encode_plus(
        X.tolist(), max_length=max_len, truncation=True, padding='max_length')
    X, mask = X['input_ids'], X['attention_mask']
    X = torch.tensor(np.array(X))
    mask = torch.tensor(np.array(mask))
    y = torch.tensor(np.array(y, dtype=int))
    data = torch.utils.data.TensorDataset(X, mask, y)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle)
    return data_loader
