# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 14:00
# @Author  : keevinzha
# @File    : txt_generator.py


import os
import pydicom
from os import path


data_path = '/Users/kevinguo/Documents/MTR_001/'

def main():
    data_dict = {}
    for i in range(3):
        for dir in os.listdir(data_path):
            if dir.endswith('.dcm'): #find dcm files
                ds = pydicom.dcmread(path.join(data_path, dir))
                if i == 0:
                    if ds.SeriesDescription == 'Sag DESS Low 3132' and ds.EchoTime == 6.428:
                        data_list = []
                        data_list.append(path.join(data_path, dir))
                        data_dict[round(ds.SliceLocation,2)] = data_list
                if i == 1:
                    if ds.SeriesDescription == 'Sag DESS Low 3132' and ds.EchoTime == 34.292:
                        data_list = data_dict[round(ds.SliceLocation,2)]
                        data_list.append(path.join(data_path, dir))
                        data_dict[round(ds.SliceLocation,2)] = data_list
                if i == 2:
                    if ds.SeriesDescription == 'NOT DIAGNOSTIC: DESS T2 map [ms]':
                        data_list = data_dict[round(ds.SliceLocation,2)]
                        data_list.append(path.join(data_path, dir))
                        data_dict[round(ds.SliceLocation,2)] = data_list


    with open("../train.txt", 'w') as f:
        for value in data_dict.values():
            f.write(" ".join(value)+'\n')
    f.close()

if __name__ == '__main__':
    main()