import numpy as np 
import math
#rnd numbers
from random import seed
from random import randint

import torch

class DataSampler(object):
    def __init__(self):
        seed(1337)

    def sample_data_and_label(data_loader, index):
        return data_loader[index]
    
    def sample_rnd_data(self, data_loader, range):
        index = randint(0, range)
        return self.sample_image_w_label(data_loader, index)
    
    def sample_data_of_label(self, data_loader, label):
        data_of_label = [data for data in data_loader if data[1] == label]
        return self.sample_rnd_data(data_of_label, len(data_of_label))
    
    def sample_untargeted_data_set_old(self, data_loader, avoid_label):
        untargeted_data_set = [data for data in data_loader if data[1] != avoid_label]
        return untargeted_data_set
        
    def sample_untargeted_data_set(self, data_loader, avoid_label):
        unstructured_data = []
        for i, (xx, label) in enumerate(data_loader):
            n = list(label.size())[0]
            for y in range(0, n-1):
                print(label[y])
                #if label[y].item() != avoid_label:
                #    unstructured_data.append((xx[y], label[y]))
        return unstructured_data
    
    def unstructure_data(self, data_loader):
        unstructured_data = []
        for i, (xx, label) in enumerate(data_loader):
            n = list(label.size())[0]
            for y in range(0, n-1):
                unstructured_data.append((xx[y], label[y]))
        return unstructured_data
    
    def sample_nearest_neighboor_old(udata, source):
        min = math.inf
        min_data
    
        for d in data:
            temp = torch.norm(source[0] - d[0][0])
            if temp < min:
                min = temp
                min_data = d
        
        return min_data
        
    def sample_nearest_neighboor(self, unstructured_data, source_x, label):
        min = math.inf
        min_x = torch.ones_like(source_x)
        min_label = torch.ones_like(label)
        if list(source_x.size())[1] == 784:
            is_flat = 1
        else:
            is_flat = 0

        for y, source in enumerate(source_x):
            for udata in unstructured_data:
                temp = torch.norm(source[0] - udata[0])
                if temp < min and udata[1].item() != label[y].item():
                    min = temp
                    if is_flat:
                        min_x[y] = udata[0].flatten()
                    else:
                        min_x[y] = udata[0]
                    min_label[y] = udata[1]
        return (min_x, min_label)
    
    def sample_adv_untargeted(self, data_loader, source, label):
        unstructured_dataset = self.unstructure_data(data_loader)
        return self.sample_nearest_neighboor(unstructured_dataset, source, label)
    
    def sample_adv_untargeted_quick(self, udata, source, label):
        return self.sample_nearest_neighboor(udata, source, label)
    
    def quick_sample_udata(self, x_batch, labels, num_labels):
        # quickly sample 1 of each label (10)
        collected_labels = set()
        unstructured_data = []
        label_counter = 0
        for i, x in enumerate(x_batch):
            if label_counter == num_labels:
                break
            l = labels[i].item()
            if l not in collected_labels:
                collected_labels.add(l)
                unstructured_data.append((x, labels[i]))
                label_counter = label_counter + 1
        return unstructured_data