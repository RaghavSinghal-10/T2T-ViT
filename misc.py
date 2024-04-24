import torch
from torch.utils.data import DataLoader, random_split
import math

def get_current_timestamp():
    return strftime("%y%m%d_%H%M%S")

def get_in_channels(data_code):
    in_ch = -1
    if data_code == 'mnist':
        in_ch = 1
    elif data_code == 'cifar10':
        in_ch = 3
    elif data_code == 'fmnist':
        in_ch = 1
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_ch

def get_in_dimensions(data_code):
    in_dim = -1    
    if data_code == 'mnist':
        in_dim = 784
    elif data_code == 'cifar10':
        in_dim = 1024
    elif data_code == 'fmnist':
        in_dim = 784
    else:
        raise ValueError("Invalid or not supported dataset [{}]".format(data_code))
    return in_dim

def get_accuracy_epoch(model, dataloader):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    acc = []
    loss = []
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    model = model.to('cuda')
    device = next(model.parameters()).device
    if getattr(model, "name", None) is None or getattr(model, "name", None) == "linear":
      for batch_idx, (data, target) in enumerate(dataloader):
          data = data.to(device)
          target = target.to(device)
          output, hiddens = model(data)
          loss.append(cross_entropy_loss(output, target).cpu().detach().numpy())
          acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())
    elif model.name and model.name=="linear-parallel":
      hiddens = {}
      for batch_idx, (data, target) in enumerate(dataloader):
          data = data.to(device)
          target = target.to(device)
          if batch_idx not in hiddens:
            hiddens[batch_idx] = []
          # output, hiddens = model(data.to(next(model.parameters()).device))
          output, hiddens[batch_idx] = model.parallel_forward(data, hiddens[batch_idx],0)
          loss.append(cross_entropy_loss(output, target).cpu().detach().numpy())
          acc.append(get_accuracy(output, target)[0].cpu().detach().numpy())
      
    return np.mean(acc), np.mean(loss)

def get_accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].float().reshape(-1).sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_accuracy_hsic(model, dataloader,config_dict={}):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []

    if model.name==None or model.name!="linear-parallel":
            # data_fraction = 0.1
      # subset_size = int(len(dataloader.dataset) * data_fraction)
      # # Split the data into a subset and the rest
      # subset_data, _ = random_split(dataloader.dataset, [subset_size, len(dataloader.dataset) - subset_size])
      
      # # Create a new data loader with the subset data
      # subset_data_loader = DataLoader(subset_data, batch_size=32, shuffle=True)
        
      # print("linear original data length",len(dataloader.dataset))
      # print("subset data length",len(subset_data_loader.dataset))
      for batch_idx, (data, target) in enumerate(dataloader):
          output, hiddens = model(data.to(next(model.parameters()).device))
          output = output.cpu().detach().numpy()
          target = target.cpu().detach().numpy().reshape(-1,1)
          output_list.append(output)
          target_list.append(target)

    elif model.name and model.name=="linear-parallel":
      # data_fraction = 0.1
      # subset_size = int(len(dataloader.dataset) * data_fraction)
      # # Split the data into a subset and the rest
      # subset_data, _ = random_split(dataloader.dataset, [subset_size, len(dataloader.dataset) - subset_size])
      
      # # Create a new data loader with the subset data
      # subset_data_loader = DataLoader(subset_data, batch_size=32, shuffle=True)

      # print("original data length",len(dataloader.dataset))
      # print("subset data length",len(subset_data_loader.dataset))
      hiddens = {}
      for batch_idx, (data, target) in enumerate(dataloader):
          # output, hiddens = model(data.to(next(model.parameters()).device))
          dev = next(model.parameters()).device
          data = data.to(dev)
          try:
            hiddens = hiddens.to(dev)
          except:
            pass
          if batch_idx not in hiddens:
            hiddens[batch_idx] = []
          output, hiddens[batch_idx] = model.parallel_forward(data, hiddens[batch_idx],0)
          del hiddens[batch_idx]
          output = output.cpu().detach().numpy()
          target = target.cpu().detach().numpy().reshape(-1,1)
          output_list.append(output)
          target_list.append(target)

    output_arr = np.vstack(output_list)
    target_arr = np.vstack(target_list)

    torch.cuda.empty_cache()

    if target_arr.shape[0] < output_arr.shape[0]:
      num_padding = output_arr.shape[0] - target_arr.shape[0]
      padding = np.zeros((num_padding, target_arr.shape[1]))
      target_arr = np.vstack((target_arr, padding))
    elif target_arr.shape[0] > output_arr.shape[0]:
      num_padding = target_arr.shape[0] - output_arr.shape[0]
      padding = np.zeros((num_padding, output_arr.shape[1]))
      output_arr = np.vstack((output_arr, padding))

    avg_acc = 0
    reorder_list = []
    for i in range(10):
        indices = np.where(target_arr==i)[0]
        select_item = output_arr[indices]
        # print("select_item",select_item.shape)
        # exit()
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        while np.argmax(y) in reorder_list:
            y[np.argmax(y)] = 0
        reorder_list.append(np.argmax(y))
        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0])/float(out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.

    return avg_acc*100., reorder_list

def get_accuracy_hsic_dp(model, dataloader,config_dict={}):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    output_list = []
    target_list = []
    # data_fraction = 0.1
    # subset_size = int(len(dataloader.dataset) * data_fraction)
    # # Split the data into a subset and the rest
    # subset_data, _ = random_split(dataloader.dataset, [subset_size, len(dataloader.dataset) - subset_size])
    
    # # Create a new data loader with the subset data
    # subset_data_loader = DataLoader(subset_data, batch_size=32, shuffle=True)
    
    # print("linear original data length",len(dataloader.dataset))
    # print("subset data length",len(subset_data_loader.dataset))
    for batch_idx, (data, target) in enumerate(dataloader):
        output, hiddens = model(data.to(next(model.parameters()).device))
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy().reshape(-1,1)
        output_list.append(output)
        target_list.append(target)


    output_arr = np.vstack(output_list)
    target_arr = np.vstack(target_list)
    
    torch.cuda.empty_cache()

    if target_arr.shape[0] < output_arr.shape[0]:
      num_padding = output_arr.shape[0] - target_arr.shape[0]
      padding = np.zeros((num_padding, target_arr.shape[1]))
      target_arr = np.vstack((target_arr, padding))
    elif target_arr.shape[0] > output_arr.shape[0]:
      num_padding = target_arr.shape[0] - output_arr.shape[0]
      padding = np.zeros((num_padding, output_arr.shape[1]))
      output_arr = np.vstack((output_arr, padding))

    avg_acc = 0
    reorder_list = []
    for i in range(10):
        indices = np.where(target_arr==i)[0]
        select_item = output_arr[indices]
        out = np.array([np.argmax(vec) for vec in select_item])
        y = np.mean(select_item, axis=0)
        while np.argmax(y) in reorder_list:
            y[np.argmax(y)] = 0
        reorder_list.append(np.argmax(y))
        num_correct = np.where(out==np.argmax(y))[0]
        accuracy = float(num_correct.shape[0])/float(out.shape[0])
        avg_acc += accuracy
    avg_acc /= 10.

    return avg_acc*100., reorder_list

# def get_accuracy_hsic_parallel(model, dataloader):
#     """ Computes the precision@k for the specified values of k
#         https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     output_list = []
#     target_list = []
#     for batch_idx, (data, target) in enumerate(dataloader):
#         # output, hiddens = model(data.to(next(model.parameters()).device))
#         output, hiddens[batch_idx] = model.parallel_forward(data.to('cuda:0'), [],0)
#         output = output.cpu().detach().numpy()
#         target = target.cpu().detach().numpy().reshape(-1,1)
#         output_list.append(output)
#         target_list.append(target)
#     output_arr = np.vstack(output_list)
#     target_arr = np.vstack(target_list)
#     avg_acc = 0
#     reorder_list = []
#     for i in range(10):
#         indices = np.where(target_arr==i)[0]
#         select_item = output_arr[indices]
#         out = np.array([np.argmax(vec) for vec in select_item])
#         y = np.mean(select_item, axis=0)
#         while np.argmax(y) in reorder_list:
#             y[np.argmax(y)] = 0
#         reorder_list.append(np.argmax(y))
#         num_correct = np.where(out==np.argmax(y))[0]
#         accuracy = float(num_correct.shape[0])/float(out.shape[0])
#         avg_acc += accuracy
#     avg_acc /= 10.

#     return avg_acc*100., reorder_list

def get_layer_parameters(model, idx_range):

    param_out = []
    param_out_name = []
    for it, (name, param) in enumerate(model.named_parameters()):
        if it in idx_range:
            param_out.append(param)
            param_out_name.append(name)

    return param_out, param_out_name


def to_categorical(y, num_classes, device='cuda'):
    """ 1-hot encodes a tensor and moves it to the specified device (default is 'cuda') """
    one_hot = torch.eye(num_classes, device=device)[y]
    return torch.squeeze(one_hot)