# imports
from __future__ import print_function, division
import numpy as np
# import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import torch.nn.functional as F
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
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
import argparse
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def setup_folders():
    # setup folders.
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("pickles", exist_ok=True)


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
#             nn.Conv1d(out_ch, out_ch, 9, padding=3, stride=1),
#             nn.ReLU(inplace=True)
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

def convert_tool_data(surfacelines):
    profile_dict = {}
    surfaceline_dict_total = {}
    for source_surfacelines in surfacelines:
        surfaceline_dict = {}

        # read the coordinates and collect to surfaceline_dict
        with open(source_surfacelines) as csvfile:
            surfacereader = csv.reader(csvfile, delimiter=';', quotechar='|')
            header = next(surfacereader)
            print("header: {}".format(header)) # not very useful
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


        # transform the data to a usable format, save to profile_dict
        for location in surfaceline_dict.keys():
            heights = np.array(surfaceline_dict[location])[:,2].astype(np.float32)

            z_tmp = np.zeros(352)
            profile_length = len(heights)
            if profile_length < 352:
                z_tmp[:profile_length] = np.array(heights, dtype=np.float32)[:profile_length]
                z_tmp[profile_length:] = heights[profile_length-1]
                heights = z_tmp
            else:
                heights = heights[:352]

            profile_dict[location] = {}
            profile_dict[location]['profile'] = heights.astype(np.float32)
            # profile_dict[location]['label'] = labels.astype(np.int32)

        for key, value in surfaceline_dict.items():
            surfaceline_dict_total[key] = value
    
    return profile_dict, surfaceline_dict_total



def main(args):
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


    # TODO next lines in function

    # path to the surfacelines and characteristic points files.
    surfacelines_path = args.profiles

    # names for the new pickles, can leave like this to overwrite each time if you just need the csv result.
    filename_dijkprofile_test_dict = 'pickles/test_dijkprofile_dict.pik'
    filename_surfaceline_test_dict = 'pickles/test_surfaceline_dict.pik'

    surfaceline_list = [
        (surfacelines_path)
    ]

    profile_dict_test, surfaceline_dict_test = convert_tool_data(surfaceline_list)
    outfile = open(filename_dijkprofile_test_dict, 'wb')
    pickle.dump(profile_dict_test, outfile)
    outfile.close()

    outfile2 = open(filename_surfaceline_test_dict, 'wb')
    pickle.dump(surfaceline_dict_test, outfile2)
    outfile2.close()

    # scale the new profiles.
    scaler = joblib.load("pickles/scaler.pik") 
    for i, key in enumerate(profile_dict_test.keys()):
        profile_dict_test[key]['profile'] = scaler.transform(profile_dict_test[key]['profile'].reshape(-1,1)).reshape(-1)
        
    print("done!")

    # TODO next lines in function

    device = "cuda:0"
    # device = "cpu:0"

    # PATH = "models/model_2019-07-24T16:32_95_dijknet.pt"
    # PATH = "models/model_2019-07-24T16:57_95_PLUS_dijknet.pt"
    PATH = args.modelpath

    # construct network
    model = Dijknet(1,17)
    # load trained weights into network
    model.load_state_dict(torch.load(PATH))
    # set network to inference/eval mode
    model.eval()
    # copy network to device
    model.to(device)
    print("loaded model!")

    # TODO next lines in function

    # final code
    inference_profile_dict = profile_dict_test
    inference_surfacelines = surfaceline_dict_test
    output_csv_basename = "data/charpoints_generated_12-2"

    model.eval()

    accumulator = np.zeros((len(inference_profile_dict), 352))
    for i, key in enumerate(inference_profile_dict.keys()):
        accumulator[i] = inference_profile_dict[key]['profile'][:352]
        
    print("total profiles to predict: ", accumulator.shape[0])
    accumulator = accumulator.reshape(822,1,352)

    outputs = model(torch.tensor(accumulator).to(device).float())
    flat_output = torch.argmax(outputs, dim=1).cpu()
    predictions = flat_output.numpy()



    header = ["LOCATIONID", "X_Maaiveld binnenwaarts", "Y_Maaiveld binnenwaarts", "Z_Maaiveld binnenwaarts", "X_Insteek sloot polderzijde", "Y_Insteek sloot polderzijde", "Z_Insteek sloot polderzijde", "X_Slootbodem polderzijde", "Y_Slootbodem polderzijde", "Z_Slootbodem polderzijde", "X_Slootbodem dijkzijde", "Y_Slootbodem dijkzijde", "Z_Slootbodem dijkzijde", "X_Insteek sloot dijkzijde", "Y_Insteek sloot dijkzijde", "Z_Insteek sloot dijkzijde", "X_Teen dijk binnenwaarts", "Y_Teen dijk binnenwaarts", "Z_Teen dijk binnenwaarts", "X_Kruin binnenberm", "Y_Kruin binnenberm", "Z_Kruin binnenberm", "X_Insteek binnenberm", "Y_Insteek binnenberm", "Z_Insteek binnenberm", "X_Kruin binnentalud", "Y_Kruin binnentalud", "Z_Kruin binnentalud", "X_Verkeersbelasting kant binnenwaarts", "Y_Verkeersbelasting kant binnenwaarts", "Z_Verkeersbelasting kant binnenwaarts", "X_Verkeersbelasting kant buitenwaarts", "Y_Verkeersbelasting kant buitenwaarts", "Z_Verkeersbelasting kant buitenwaarts", "X_Kruin buitentalud", "Y_Kruin buitentalud", "Z_Kruin buitentalud", "X_Insteek buitenberm", "Y_Insteek buitenberm", "Z_Insteek buitenberm", "X_Kruin buitenberm", "Y_Kruin buitenberm", "Z_Kruin buitenberm", "X_Teen dijk buitenwaarts", "Y_Teen dijk buitenwaarts", "Z_Teen dijk buitenwaarts", "X_Insteek geul", "Y_Insteek geul", "Z_Insteek geul", "X_Teen geul", "Y_Teen geul", "Z_Teen geul", "X_Maaiveld buitenwaarts", "Y_Maaiveld buitenwaarts", "Z_Maaiveld buitenwaarts"]

    # construct entries
    with open('{}.csv'.format(output_csv_basename), 'w') as csvFile:
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
                    if index >= len(surfaceline_dict_test[key]):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", help="input surfaceline csv.", required=True)
    parser.add_argument("--outname", help="outputname for the annotation file.", default="data/charpoints_scriptgenerated_12-2")
    parser.add_argument("--modelpath", help="path to the annotation model file.", default="models/model_2019-09-19T14:34_95p_fryslan_dijknet.pt")
    args = parser.parse_args()
    main(args)

    