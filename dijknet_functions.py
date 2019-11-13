# imports
from __future__ import print_function, division
import numpy as np
from IPython.core.debugger import set_trace
import torch.nn.functional as F
import csv
import torch
import torch.nn as nn
import time
import pickle
from sklearn.preprocessing import StandardScaler
import os
import torch
import pandas as pd
import glob
import joblib
import seaborn as sns
import datetime 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter
from torch.autograd import Variable
from PIL import Image
from torch.utils import data
from sklearn.model_selection import train_test_split

ben_conversion_dict = {
    "": "leeg",
    "101_Q19_2" : "buitenkruin",
    "101_Q19_3" : "binnenkruin",
    "101_Q19_5" : "binnenteen",
    "105_T09_11": "insteek_sloot",
    "811_T13_8" : "leeg",
    "351_T03_10" : "leeg",
    "_T01_KKW" : "leeg",
    "108_Q06_250" : "leeg",
    "303_Q05_1": "leeg",
    "353__11" : "leeg",
    "_T00_17" : "leeg",
    "109_Q08_13" : "leeg",
    "_Q07_KDM" : "leeg",
    "_Q07_KDW" : "leeg",
    '0' : "leeg",
    None : "leeg",
    'nan' : "leeg"
}
class_dict_regionaal = {
    "leeg": 0,
    "startpunt": 1,
    "buitenkruin": 2,
    "binnenkruin": 3,
    "binnenteen": 4,
    "insteek_sloot": 5
}

class_dict = {
    'leeg': 0,
    'Maaiveld binnenwaarts': 1,
    'Insteek sloot polderzijde': 2,
    'Slootbodem polderzijde': 3,
    'Slootbodem dijkzijde': 4,
    'Insteek sloot dijkzijde': 5,
    'Teen dijk binnenwaarts': 6,
    'Kruin binnenberm': 7,
    'Insteek binnenberm': 8,
    'Kruin binnentalud': 9,
    'Verkeersbelasting kant binnenwaarts': 9, # 10
    'Verkeersbelasting kant buitenwaarts': 10,
    'Kruin buitentalud': 10, #12
    'Insteek buitenberm': 11,
    'Kruin buitenberm': 12,
    'Teen dijk buitenwaarts': 13,
    'Insteek geul': 14,
    'Teen geul': 15,
    'Maaiveld buitenwaarts': 16,
}

# set the class dict to either the full class_dict for non-regional keringen (leave it)
# or set it to the smaller dict for regional keringen by uncommenting this line

# class_dict = class_dict_regionaal
inverse_class_dict = {v: k for k, v in class_dict.items()}

# manual mappings to get the correct names for plotting later
if "11" in inverse_class_dict:
    inverse_class_dict["10"] = 'Kruin buitentalud'
    

class Double_conv(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''
    def __init__(self, in_ch, out_ch, p):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 7, padding=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_down(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''
    
    def __init__(self, in_ch, out_ch, p):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.conv = Double_conv(in_ch, out_ch, p)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x


class Conv_up(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''
    
    def __init__(self, in_ch, out_ch, p):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up, self).__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = Double_conv(in_ch, out_ch, p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


def extract_img(size, in_tensor):
    """
    Args:
        size(int) : size of cut
        in_tensor(tensor) : tensor to be cut
    """
    dim1 = in_tensor.size()[2]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((size + (dim1-size)/2))]
    return in_tensor


class Dijknet(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.25):
        super(Dijknet, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 64, p)
        self.Conv_down2 = Conv_down(64, 128, p)
        self.Conv_down3 = Conv_down(128, 256, p)
        self.Conv_down4 = Conv_down(256, 512, p)
        self.Conv_down5 = Conv_down(512, 1024, p)
        self.Conv_up1 = Conv_up(1024, 512, p)
        self.Conv_up2 = Conv_up(512, 256, p)
        self.Conv_up3 = Conv_up(256, 128, p)
        self.Conv_up4 = Conv_up(128, 64, p)
        self.Conv_up5 = Conv_up(128, 64, p)
        self.Conv_out = nn.Conv1d(64, out_channels, 1, padding=0, stride=1)
        self.Conv_final = nn.Conv1d(out_channels, out_channels, 1, padding=0, stride=1)

    def forward(self, x):
        input_image = x
        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        x, conv4 = self.Conv_down4(x)
        _, x = self.Conv_down5(x)
        x = self.Conv_up1(x, conv4)
        x = self.Conv_up2(x, conv3)
        x = self.Conv_up3(x, conv2)
        x = self.Conv_up4(x, conv1)
        # final upscale to true size
        x = self.Conv_out(x)
        x = self.Conv_final(x)
        return x
    
def ffill(arr):
    """Forward fill utility function."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def convert_tool_data(annotation_tuples):
    profile_dict = {}
    surfaceline_dict_total = {}
    for source_surfacelines, source_characteristicpoints in annotation_tuples:
        surfaceline_dict = {}

        # read the coordinates and collect to surfaceline_dict
        with open(source_surfacelines) as csvfile:
            surfacereader = csv.reader(csvfile, delimiter=';', quotechar='|')
            header = next(surfacereader)
            # print("header: {}".format(header)) # not very useful
            for row in surfacereader:
                location = row[0]
                surfaceline_dict[location] = []
                for i in range(1, len(row)-2, 3):
                    try:
                        x = float(row[i])
                        y = float(row[i+1])
                        z = float(row[i+2])
                        surfaceline_dict[location].append((x,y,z))
                    except:
                        print("error reading point from surfaceline at location: ", location)
                        print("index: ", i)
                        pass

        print("loaded surfacelines for {} locations".format(len(surfaceline_dict.keys())))

        cpoints_dict = {}
        # read the characteristicpoints and save to cpoints_dict
        with open(source_characteristicpoints) as csvfile:
            cpointsreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            header = next(cpointsreader)
            for row in cpointsreader:
                location = row[0]
                point_dict = {}
                for i in range(1, len(row)-2, 3):
                    try:
                        x = float(row[i])
                        y = float(row[i+1])
                        z = float(row[i+2])

                        point_dict[header[i][2:]] = (x,y,z)
                    except:
                        pass
                cpoints_dict[location] = point_dict

        print("loaded characteristic points for {} locations".format(len(cpoints_dict.keys())))

        # transform the data to a usable format, save to profile_dict
        X_samples_list = []
        Y_samples_list = []
        location_list = []
        for location in surfaceline_dict.keys():
            heights = np.array(surfaceline_dict[location])[:,2].astype(np.float32)
            x_y_s = np.array(surfaceline_dict[location])[:,:2].astype(np.float32)
            labels = np.zeros(len(heights))
            for i, (key, point) in enumerate(cpoints_dict[location].items()):
                # if the point is not empty, find the nearest point in the surface file, 
                # rounding errors require matching by distance per point
                if point == (-1.0, -1.0, -1.0):
                    continue

                distances = []
                for idx, surfacepoint in enumerate(surfaceline_dict[location]):
                    dist = np.linalg.norm(np.array(surfacepoint)-np.array(point))
                    distances.append((idx, dist))
                (idx, dist) = sorted(distances, key=itemgetter(1))[0]
                labels[idx] = class_dict[key]

            for i in range(1, len(labels)):
                if labels[i] == 0.0:
                    labels[i] = labels[i-1]

            z_tmp = np.zeros(352)
            labels_tmp = np.zeros(352)
            profile_length = labels.shape[0]
            if profile_length < 352:
                z_tmp[:profile_length] = np.array(heights, dtype=np.float32)[:profile_length]
                labels_tmp[:profile_length] = np.array(labels)[:profile_length]
                z_tmp[profile_length:] = heights[profile_length-1]
                labels_tmp[profile_length:] = labels[profile_length-1]
                heights = z_tmp
                labels = labels_tmp
            else:
                heights = heights[:352]
                labels = labels[:352]

            profile_dict[location] = {}
            profile_dict[location]['profile'] = heights.astype(np.float32)
            profile_dict[location]['label'] = labels.astype(np.int32)

        for key, value in surfaceline_dict.items():
            surfaceline_dict_total[key] = value
    
    return profile_dict, surfaceline_dict_total

class DijkprofileDataset(data.Dataset):
    """Pytorch custom dataset class to use with the pytorch dataloader."""
    
    def __init__(self, profile_dict, partition):
        'init'
        self.data_dict = profile_dict
        self.list_IDs = partition
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = self.data_dict[ID]['profile'].reshape(1,-1).astype(np.float32)
        y = self.data_dict[ID]['label'].reshape(1,-1)
        return X, y
    
    def __str__(self):
        return "<Dijkprofile dataset: datapoints={}>".format(len(self.list_IDs))


def get_loss_train(model, data_train, criterion):
    """Calculate loss over train set"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_acc = 0
    total_loss = 0
    for batch, (images, masks) in enumerate(data_train):
        with torch.no_grad():
            images = Variable(images.to(device))
            masks = Variable(masks.to(device))
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1).float()
            acc = accuracy_check_for_batch(masks.cpu(), preds.cpu(), images.size()[0])
            total_acc = total_acc + acc
            total_loss = total_loss + loss.cpu().item()
    return total_acc/(batch+1), total_loss/(batch + 1)

def make_annotations(inference_profile_dict, inference_surfacelines, model, output_csv_basename, device):
    model.eval()
    accumulator = np.zeros((len(inference_profile_dict), 352))
    for i, key in enumerate(inference_profile_dict.keys()):
        accumulator[i] = inference_profile_dict[key]['profile'][:352]

    print("total profiles to predict: ", accumulator.shape[0])
    accumulator = accumulator.reshape(accumulator.shape[0],1,352)

    outputs = model(torch.tensor(accumulator).to(device).float())
    flat_output = torch.argmax(outputs, dim=1).cpu()
    predictions = flat_output.numpy()

    header = ["LOCATIONID", "X_Maaiveld binnenwaarts", "Y_Maaiveld binnenwaarts", "Z_Maaiveld binnenwaarts", "X_Insteek sloot polderzijde", "Y_Insteek sloot polderzijde", "Z_Insteek sloot polderzijde", "X_Slootbodem polderzijde", "Y_Slootbodem polderzijde", "Z_Slootbodem polderzijde", "X_Slootbodem dijkzijde", "Y_Slootbodem dijkzijde", "Z_Slootbodem dijkzijde", "X_Insteek sloot dijkzijde", "Y_Insteek sloot dijkzijde", "Z_Insteek sloot dijkzijde", "X_Teen dijk binnenwaarts", "Y_Teen dijk binnenwaarts", "Z_Teen dijk binnenwaarts", "X_Kruin binnenberm", "Y_Kruin binnenberm", "Z_Kruin binnenberm", "X_Insteek binnenberm", "Y_Insteek binnenberm", "Z_Insteek binnenberm", "X_Kruin binnentalud", "Y_Kruin binnentalud", "Z_Kruin binnentalud", "X_Verkeersbelasting kant binnenwaarts", "Y_Verkeersbelasting kant binnenwaarts", "Z_Verkeersbelasting kant binnenwaarts", "X_Verkeersbelasting kant buitenwaarts", "Y_Verkeersbelasting kant buitenwaarts", "Z_Verkeersbelasting kant buitenwaarts", "X_Kruin buitentalud", "Y_Kruin buitentalud", "Z_Kruin buitentalud", "X_Insteek buitenberm", "Y_Insteek buitenberm", "Z_Insteek buitenberm", "X_Kruin buitenberm", "Y_Kruin buitenberm", "Z_Kruin buitenberm", "X_Teen dijk buitenwaarts", "Y_Teen dijk buitenwaarts", "Z_Teen dijk buitenwaarts", "X_Insteek geul", "Y_Insteek geul", "Z_Insteek geul", "X_Teen geul", "Y_Teen geul", "Z_Teen geul", "X_Maaiveld buitenwaarts", "Y_Maaiveld buitenwaarts", "Z_Maaiveld buitenwaarts"]

    # construct entries
    with open(output_csv_basename, 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for i, key in enumerate(inference_profile_dict.keys()):
            # get predictions
            profile_pred = predictions[i]

            # construct dict with key for each row
            row_dict = {key:-1 for key in header}
            row_dict["LOCATIONID"] = key

            # loop through predictions and for the entries
            used_classes = []
            prev_class_n = 999 # key thats not in the inverse_class_dict
            for index, class_n in enumerate(profile_pred):
                if class_n == 0 or class_n in used_classes:
                    continue
                if class_n != prev_class_n:
                    # get class name
                    class_name = inverse_class_dict[class_n]

                    # if this index is different from the last, this is the characteristicpoint
                    used_classes.append(prev_class_n)

                    # set prev_class to the new class
                    prev_class_n = class_n

                    # construct the csv row with the new class
                    if index >= len(inference_surfacelines[key]):
                        continue

                    (x,y,z) = inference_surfacelines[key][index]
                    row_dict["X_" + class_name] = round(x, 3)
                    row_dict["Y_" + class_name] = round(y, 3)
                    row_dict["Z_" + class_name] = round(z, 3)

            # write the row to the csv file
            row = []
            for columnname in header:
                row.append(row_dict[columnname])
            writer.writerow(row)

    csvFile.close()
    print("done!")