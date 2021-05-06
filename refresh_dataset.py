import math
import os
import random
import shutil
import json
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from PIL import ImagePath

###############################################################################
############################         config         ###########################
###############################################################################

CLIPS_ROOT_FOLDER = "/mnt/g27prist/TCO/TCO-Studenten/bleeding/Clips"

ANNOTATION_XLSX_PATH = "/mnt/g27prist/TCO/TCO-Studenten/bleeding/Annotation Dateien.xlsx"
DATA_PATH = "/mnt/g27prist/TCO/TCO-Studenten/zhaotian/new_dataset"
DATA_COPY_TO = '/mnt/g27prist/TCO/TCO-Studenten/zhaotian/grouped_dataset'
DATA_GAP = 1        # pick a frame as data for each DATA_GAP frames


NAME_MATCHING = {
    '2019_08_12_anterioreRektumresektion': 'ARR_190812',
    '2019_08_13_abdominothorakaleÖsophagusresektion_2': 'AOOR190813',
    '2019_09_19_abdominothorakaleÖsophagusresektion': 'AOOR190819',
    '06082019abdominothorakaleOesophagusresektion2': 'AOOR190806',
    '20191029_oesophagusresektion1': 'OR__191029',
    '19082019tiefeRektumresektion': 'TRR_190819',
    '28082019Gastrektomie': 'GE__190828',
    'oesophagusresektion': 'OR__xxxxxx',
    'part1': 'P1__xxxxxx',
    'part2': 'P2__xxxxxx'
}


###############################################################################
############################       functions        ###########################
###############################################################################

class MaskCreater:
    """Create all masks for the images under given ori_folder with help from a csv data, which contains the polygon position data.

        To use:
            mc = MaskCreater(
                ori_folder = './2019_08_13_abdominothorakaleÖsophagusresektion_2_00.45.38-00.58.30', 
                mask_folder = './2019_08_13_abdominothorakaleÖsophagusresektion_2_00.45.38-00.58.30_mask', 
                csv_path = '2019_08_13_abdominothorakaleÖsophagusresektion_2_00.45.38-00.58.30.csv'
                )

            mc.process_ori_folder()
        
        Then the black-white masks are saved under given mask_folder, as .jpg with format RGB

     """

    def __init__(self, *, ori_folder, mask_folder, csv_path):
        self.ori_folder = ori_folder
        self.mask_folder = mask_folder
        self.csv_path = csv_path
        self.naming_method = lambda ori_name : ori_name + '_mask'

        if not os.path.exists(self.mask_folder):
            os.makedirs(self.mask_folder)

    def genrate_mask_img(self, *, ori_name, polygons):

        # get size of origional image
        im = Image.open(os.path.join(self.ori_folder, ori_name))
        width, height = im.size

        # create new mask
        img = Image.new("RGB", (width, height))
        img1 = ImageDraw.Draw(img)

        # if x y positions are not empty, draw polygon
        for polygon in polygons:
            x_pos = polygon['x_pos']
            y_pos = polygon['y_pos']

            # return false if given point is not in image area 
            if max(x_pos) > width or max(y_pos) > height:
                raise ValueError('Given position is in outside of the image')

            # draw black white image in RGB mode
            xy = [ (x_pos[i], y_pos[i]) for i in range(len(x_pos))]
            img1.polygon(xy, fill ="white")

        clip_name = self.ori_folder.split('/')[-1] + '_'

        # save mask image
        filename, file_extension = os.path.splitext(ori_name)
        img.save(os.path.join(self.mask_folder, clip_name + self.naming_method(filename) + file_extension))

        im_path = os.path.join(self.ori_folder, ori_name)
        im_copy_path = os.path.join(self.mask_folder, clip_name + ori_name)
        shutil.copyfile(im_path, im_copy_path)

    def process_ori_folder(self):

        csv_df = pd.read_csv(self.csv_path)
        csv_df = csv_df.set_index('filename')

        print("Processing csv: {}".format(self.csv_path))
        num_img = 0

        for img_name in os.listdir(self.ori_folder):

            shape_attributes_str = csv_df.at[img_name, 'region_shape_attributes']

            if len(str(shape_attributes_str)) > 2:  #len('{}') = 2

                polygons = []

                if type(shape_attributes_str) == np.ndarray:
                    polygons = list( {'x_pos': json.loads(shape_attributes)['all_points_x'], 'y_pos': json.loads(shape_attributes)['all_points_y']} for shape_attributes in shape_attributes_str if json.loads(shape_attributes)['name'] == "polygon" )
                else:
                    shape_attributes = json.loads(shape_attributes_str)
                    polygons.append({'x_pos': shape_attributes['all_points_x'], 'y_pos': shape_attributes['all_points_y']})

                if len(polygons) == 0:
                    print("Longer than 2 but no polygon:" + str(shape_attributes_str))

                self.genrate_mask_img(ori_name = img_name, polygons = polygons)
                num_img = num_img + 1
            else:
                if str(shape_attributes_str) != '{}':
                    print("smaller than 2 but not empty:" + str(shape_attributes_str))


        print("Total {} images".format(num_img))


def read_clips(xlsx_path=''):
    """
    Each clip will be trited as a group. A csv file should exist if the lable 
    in annotation.xlsx under column "has_valid_csv" is 1.
    """

    assert bool(xlsx_path)

    xlsx = pd.read_excel(xlsx_path)
    annotation_status_series = xlsx["has_valid_csv"].astype(str).str.contains('1').fillna(False)
    clip_title_list = list(xlsx[annotation_status_series]["clip"])
    op_title_list = list(xlsx[annotation_status_series]["OP"])

    clip_list = []
    for index in range(len(clip_title_list)):

        root_folder_path = CLIPS_ROOT_FOLDER + "/" + op_title_list[index]

        clip = {
            "op_title": op_title_list[index],
            "clip_title": clip_title_list[index],
            "img_folder": root_folder_path + "/" + clip_title_list[index],
            "mp4_path"  : root_folder_path + "/" + clip_title_list[index] + ".MP4",
            "csv_path"  : root_folder_path + "/via_export_csv" + clip_title_list[index] + ".csv"
        }

        clip_list.append(clip)
    
    # check if all required folder and csv-file exist
    for obj in clip_list:
        try:
            assert os.path.isdir(obj["img_folder"])
        except AssertionError:
            print("Error: file await but not found at {}".format(obj["img_folder"]))
        """
        try:
            assert os.path.isfile(obj["mp4_path"])
        except AssertionError:
            print("Error: file await but not found at {}".format(obj["mp4_folder"]))
        """
        try:
            assert os.path.isfile(obj["csv_path"])
        except AssertionError:
            print("Error: file await but not found at {}".format(obj["csv_path"]))
            
    return clip_list

###############################################################################
############################         main          ############################
###############################################################################

clip_list = read_clips(xlsx_path=ANNOTATION_XLSX_PATH)

for index, clip in enumerate(clip_list):
    print()
    print('_' * 10)
    print("Processing clip {}/{}, {}".format(index+1, len(clip_list), clip["clip_title"]))

    mc = MaskCreater(
        ori_folder = clip["img_folder"], 
        mask_folder = DATA_PATH + '/' + clip["clip_title"], 
        csv_path = clip["csv_path"]
    )

    mc.process_ori_folder()



if not os.path.isdir(DATA_COPY_TO):
    os.mkdir(DATA_COPY_TO)


folders = os.listdir(DATA_PATH)

total = 0

secure_random = random.SystemRandom()

for folder_name in folders:
    print("prossising: {}".format(folder_name))

    op_name = '_'.join(folder_name.split('_')[:-1])
    op_code = NAME_MATCHING[op_name]

    timestamp = folder_name.split('_')[-1].replace('-', '').replace('.', '')

    group_id = op_code + '_' + timestamp

    images_path = os.listdir(os.path.join(DATA_PATH, folder_name))

    pure_file_name_list = list(os.path.splitext(p.replace('_mask', ''))[0] for p in images_path)
    img_index_list = list(int(p.split('_')[-1]) for p in pure_file_name_list)
    group_number = list(int(i / DATA_GAP) for i in img_index_list)

    total += len(list(set(group_number)))

    group_matching = dict((key, []) for key in list(set(group_number)))

    for index, group_numer_value in enumerate(group_number):
        if 'mask' not in images_path[index]:
            group_matching[group_numer_value].append(os.path.join(DATA_PATH, folder_name, images_path[index]))
    
    random_picked = dict((key, []) for key in list(set(group_number)))
    for key in list(set(group_number)):
        picked = secure_random.choice(group_matching[key])
        random_picked[key] = picked

    group_index = 0
    for index, key in enumerate(list(random_picked.keys())):
        if index % 5 == 0:
            group_index += 1
        if not os.path.isdir(os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index))):
            os.mkdir(os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index)))
        shutil.copy(random_picked[key], os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index), os.path.split(random_picked[key])[1]))

        mask_name = '_mask'.join(list(os.path.splitext(random_picked[key])))
        shutil.copy(mask_name, os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index), os.path.split(mask_name)[1]))


###############################################################################
############################       functions        ###########################
###############################################################################

class NobloodMaskCreater:
    """Create all masks for the images under given ori_folder with help from a csv data, which contains the polygon position data.

        To use:
            mc = MaskCreater(
                ori_folder = './2019_08_13_abdominothorakaleÖsophagusresektion_2_00.45.38-00.58.30', 
                mask_folder = './2019_08_13_abdominothorakaleÖsophagusresektion_2_00.45.38-00.58.30_mask'
                )

            mc.process_ori_folder()
        
        Then the black-white masks are saved under given mask_folder, as .jpg with format RGB

     """

    def __init__(self, *, ori_folder, mask_folder):
        self.ori_folder = ori_folder
        self.naming_method = lambda ori_name : ori_name + '_mask'
        self.mask_folder = mask_folder

        if not os.path.exists(self.mask_folder):
            os.makedirs(self.mask_folder)

    def genrate_mask_img(self, *, ori_name):

        # get size of origional image
        im = Image.open(os.path.join(self.ori_folder, ori_name))
        width, height = im.size

        # create new mask
        img = Image.new("RGB", (width, height))
        img1 = ImageDraw.Draw(img)

        clip_name = self.ori_folder.split('/')[-1] + '_'

        # save mask image
        filename, file_extension = os.path.splitext(ori_name)
        img.save(os.path.join(self.mask_folder, clip_name + self.naming_method(filename) + file_extension))

        im_path = os.path.join(self.ori_folder, ori_name)
        im_copy_path = os.path.join(self.mask_folder, clip_name + ori_name)
        shutil.copyfile(im_path, im_copy_path)

    def process_ori_folder(self):

        for img_name in os.listdir(self.ori_folder):

            self.genrate_mask_img(ori_name = img_name)

###############################################################################
############################         main          ############################
###############################################################################

DATA_PATH = '/mnt/g27prist/TCO/TCO-Studenten/zhaotian/new_dataset_without_blood'
DATA_COPY_TO = '/mnt/g27prist/TCO/TCO-Studenten/zhaotian/grouped_dataset_no_blood'
DATA_GAP = 1        # pick a frame as data for each DATA_GAP frames

clip_list = [
    {
        "img_folder": "/mnt/g27prist/TCO/TCO-Studenten/bleeding/Clips_no_blood/2019_08_13_abdominothorakaleÖsophagusresektion_2_00.00.00-00.08.28",
        "clip_title": "2019_08_13_abdominothorakaleÖsophagusresektion_2_00.00.00-00.08.28"
    },
    {
        "img_folder": "/mnt/g27prist/TCO/TCO-Studenten/bleeding/Clips_no_blood/06082019abdominothorakaleOesophagusresektion2_00.40.02-00.53.14",
        "clip_title": "06082019abdominothorakaleOesophagusresektion2_00.40.02-00.53.14"
    },    {
        "img_folder": "/mnt/g27prist/TCO/TCO-Studenten/bleeding/Clips_no_blood/19082019tiefeRektumresektion_00.15.43-00.26.02",
        "clip_title": "19082019tiefeRektumresektion_00.15.43-00.26.02"
    },    {
        "img_folder": "/mnt/g27prist/TCO/TCO-Studenten/bleeding/Clips_no_blood/19082019tiefeRektumresektion_01.11.26-01.20.09",
        "clip_title": "19082019tiefeRektumresektion_01.11.26-01.20.09"
    },    {
        "img_folder": "/mnt/g27prist/TCO/TCO-Studenten/bleeding/Clips_no_blood/19082019tiefeRektumresektion_01.23.45-01.52.36",
        "clip_title": "19082019tiefeRektumresektion_01.23.45-01.52.36"
    }
]

for index, clip in enumerate(clip_list):
    print()
    print('_' * 10)
    print("Processing clip {}/{}, {}".format(index+1, len(clip_list), clip["clip_title"]))

    mc = NobloodMaskCreater(
        ori_folder = clip["img_folder"], 
        mask_folder = DATA_PATH + '/' + clip["clip_title"]
    )

    mc.process_ori_folder()

if not os.path.isdir(DATA_COPY_TO):
    os.mkdir(DATA_COPY_TO)

NAME_MATCHING = {
    '2019_08_12_anterioreRektumresektion': 'ARR_190812',
    '2019_08_13_abdominothorakaleÖsophagusresektion_2': 'AOOR190813',
    '2019_09_19_abdominothorakaleÖsophagusresektion': 'AOOR190819',
    '06082019abdominothorakaleOesophagusresektion2': 'AOOR190806',
    '20191029_oesophagusresektion1': 'OR__191029',
    '19082019tiefeRektumresektion': 'TRR_190819',
    '28082019Gastrektomie': 'GE__190828',
    'oesophagusresektion': 'OR__xxxxxx',
    'part1': 'P1__xxxxxx',
    'part2': 'P2__xxxxxx'
}

folders = os.listdir(DATA_PATH)

total = 0

secure_random = random.SystemRandom()

for folder_name in folders:
    print("prossising: {}".format(folder_name))

    op_name = '_'.join(folder_name.split('_')[:-1])
    op_code = NAME_MATCHING[op_name]

    timestamp = folder_name.split('_')[-1].replace('-', '').replace('.', '')

    group_id = op_code + '_' + timestamp

    images_path = os.listdir(os.path.join(DATA_PATH, folder_name))

    pure_file_name_list = list(os.path.splitext(p.replace('_mask', ''))[0] for p in images_path)
    img_index_list = list(int(p.split('_')[-1]) for p in pure_file_name_list)
    group_number = list(int(i / DATA_GAP) for i in img_index_list)

    total += len(list(set(group_number)))

    group_matching = dict((key, []) for key in list(set(group_number)))

    for index, group_numer_value in enumerate(group_number):
        if 'mask' not in images_path[index]:
            group_matching[group_numer_value].append(os.path.join(DATA_PATH, folder_name, images_path[index]))
    
    random_picked = dict((key, []) for key in list(set(group_number)))
    for key in list(set(group_number)):
        picked = secure_random.choice(group_matching[key])
        random_picked[key] = picked

    group_index = 0
    for index, key in enumerate(list(random_picked.keys())):
        if index % 5 == 0:
            group_index += 1
        if not os.path.isdir(os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index))):
            os.mkdir(os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index)))
        shutil.copy(random_picked[key], os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index), os.path.split(random_picked[key])[1]))

        mask_name = '_mask'.join(list(os.path.splitext(random_picked[key])))
        shutil.copy(mask_name, os.path.join(DATA_COPY_TO, group_id + '_' + str(group_index), os.path.split(mask_name)[1]))


