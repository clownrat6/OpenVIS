# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse BURST dataset 
annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_burst_json", "register_burst_instances"]


ALL_BURST_CATEGORIES = [
    {'color': [17, 120, 192], 'isthing': 1, 'id': 1, 'lvis_id': 2, 'name': 'aerosol_can'},
    {'color': [62, 55, 240], 'isthing': 1, 'id': 2, 'lvis_id': 4, 'name': 'airplane'},
    {'color': [136, 10, 55], 'isthing': 1, 'id': 3, 'lvis_id': 16, 'name': 'apricot'},
    {'color': [150, 221, 250], 'isthing': 1, 'id': 4, 'lvis_id': 17, 'name': 'apron'},
    {'color': [222, 247, 144], 'isthing': 1, 'id': 5, 'lvis_id': 20, 'name': 'armchair'},
    {'color': [39, 124, 64], 'isthing': 1, 'id': 6, 'lvis_id': 24, 'name': 'trash_can'},
    {'color': [56, 129, 146], 'isthing': 1, 'id': 7, 'lvis_id': 25, 'name': 'ashtray'},
    {'color': [197, 93, 3], 'isthing': 1, 'id': 8, 'lvis_id': 27, 'name': 'atomizer'},
    {'color': [175, 118, 123], 'isthing': 1, 'id': 9, 'lvis_id': 31, 'name': 'ax'},
    {'color': [157, 25, 74], 'isthing': 1, 'id': 10, 'lvis_id': 32, 'name': 'baby_buggy'},
    {'color': [6, 238, 75], 'isthing': 1, 'id': 11, 'lvis_id': 34, 'name': 'backpack'},
    {'color': [219, 21, 107], 'isthing': 1, 'id': 12, 'lvis_id': 35, 'name': 'handbag'},
    {'color': [84, 64, 224], 'isthing': 1, 'id': 13, 'lvis_id': 36, 'name': 'suitcase'},
    {'color': [160, 133, 215], 'isthing': 1, 'id': 14, 'lvis_id': 37, 'name': 'bagel'},
    {'color': [199, 148, 151], 'isthing': 1, 'id': 15, 'lvis_id': 38, 'name': 'bagpipe'},
    {'color': [252, 253, 176], 'isthing': 1, 'id': 16, 'lvis_id': 41, 'name': 'ball'},
    {'color': [105, 253, 245], 'isthing': 1, 'id': 17, 'lvis_id': 43, 'name': 'balloon'},
    {'color': [124, 102, 189], 'isthing': 1, 'id': 18, 'lvis_id': 45, 'name': 'banana'},
    {'color': [32, 226, 44], 'isthing': 1, 'id': 19, 'lvis_id': 47, 'name': 'bandage'},
    {'color': [215, 200, 7], 'isthing': 1, 'id': 20, 'lvis_id': 50, 'name': 'banner'},
    {'color': [111, 78, 165], 'isthing': 1, 'id': 21, 'lvis_id': 51, 'name': 'barbell'},
    {'color': [50, 36, 15], 'isthing': 1, 'id': 22, 'lvis_id': 55, 'name': 'barrow'},
    {'color': [92, 52, 73], 'isthing': 1, 'id': 23, 'lvis_id': 58, 'name': 'baseball_bat'},
    {'color': [252, 186, 136], 'isthing': 1, 'id': 24, 'lvis_id': 60, 'name': 'baseball_glove'},
    {'color': [97, 247, 92], 'isthing': 1, 'id': 25, 'lvis_id': 61, 'name': 'basket'},
    {'color': [116, 84, 157], 'isthing': 1, 'id': 26, 'lvis_id': 62, 'name': 'basketball_hoop'},
    {'color': [96, 236, 38], 'isthing': 1, 'id': 27, 'lvis_id': 63, 'name': 'basketball'},
    {'color': [198, 204, 32], 'isthing': 1, 'id': 28, 'lvis_id': 66, 'name': 'bath_mat'},
    {'color': [214, 100, 116], 'isthing': 1, 'id': 29, 'lvis_id': 74, 'name': 'beaker'},
    {'color': [55, 204, 131], 'isthing': 1, 'id': 30, 'lvis_id': 77, 'name': 'beanie'},
    {'color': [79, 24, 212], 'isthing': 1, 'id': 31, 'lvis_id': 78, 'name': 'bear'},
    {'color': [212, 196, 115], 'isthing': 1, 'id': 32, 'lvis_id': 79, 'name': 'bed'},
    {'color': [195, 117, 154], 'isthing': 1, 'id': 33, 'lvis_id': 80, 'name': 'bedspread'},
    {'color': [138, 174, 118], 'isthing': 1, 'id': 34, 'lvis_id': 81, 'name': 'cow'},
    {'color': [154, 91, 74], 'isthing': 1, 'id': 35, 'lvis_id': 83, 'name': 'beeper'},
    {'color': [173, 248, 35], 'isthing': 1, 'id': 36, 'lvis_id': 85, 'name': 'beer_can'},
    {'color': [174, 132, 87], 'isthing': 1, 'id': 37, 'lvis_id': 87, 'name': 'bell'},
    {'color': [185, 142, 112], 'isthing': 1, 'id': 38, 'lvis_id': 88, 'name': 'bell_pepper'},
    {'color': [142, 90, 164], 'isthing': 1, 'id': 39, 'lvis_id': 91, 'name': 'bench'},
    {'color': [130, 110, 143], 'isthing': 1, 'id': 40, 'lvis_id': 93, 'name': 'bib'},
    {'color': [63, 32, 79], 'isthing': 1, 'id': 41, 'lvis_id': 95, 'name': 'bicycle'},
    {'color': [77, 3, 232], 'isthing': 1, 'id': 42, 'lvis_id': 97, 'name': 'binder'},
    {'color': [143, 62, 116], 'isthing': 1, 'id': 43, 'lvis_id': 98, 'name': 'binoculars'},
    {'color': [159, 212, 233], 'isthing': 1, 'id': 44, 'lvis_id': 99, 'name': 'bird'},
    {'color': [125, 81, 165], 'isthing': 1, 'id': 45, 'lvis_id': 100, 'name': 'birdfeeder'},
    {'color': [195, 207, 247], 'isthing': 1, 'id': 46, 'lvis_id': 102, 'name': 'birdcage'},
    {'color': [7, 12, 190], 'isthing': 1, 'id': 47, 'lvis_id': 103, 'name': 'birdhouse'},
    {'color': [86, 176, 6], 'isthing': 1, 'id': 48, 'lvis_id': 106, 'name': 'biscuit_(bread)'},
    {'color': [180, 4, 74], 'isthing': 1, 'id': 49, 'lvis_id': 108, 'name': 'black_sheep'},
    {'color': [212, 55, 214], 'isthing': 1, 'id': 50, 'lvis_id': 110, 'name': 'blanket'},
    {'color': [134, 33, 234], 'isthing': 1, 'id': 51, 'lvis_id': 112, 'name': 'blender'},
    {'color': [210, 248, 221], 'isthing': 1, 'id': 52, 'lvis_id': 117, 'name': 'gameboard'},
    {'color': [135, 100, 147], 'isthing': 1, 'id': 53, 'lvis_id': 118, 'name': 'boat'},
    {'color': [188, 39, 125], 'isthing': 1, 'id': 54, 'lvis_id': 126, 'name': 'book'},
    {'color': [235, 146, 175], 'isthing': 1, 'id': 55, 'lvis_id': 127, 'name': 'book_bag'},
    {'color': [153, 133, 162], 'isthing': 1, 'id': 56, 'lvis_id': 129, 'name': 'booklet'},
    {'color': [9, 83, 102], 'isthing': 1, 'id': 57, 'lvis_id': 133, 'name': 'bottle'},
    {'color': [166, 116, 49], 'isthing': 1, 'id': 58, 'lvis_id': 134, 'name': 'bottle_opener'},
    {'color': [238, 240, 158], 'isthing': 1, 'id': 59, 'lvis_id': 135, 'name': 'bouquet'},
    {'color': [121, 44, 235], 'isthing': 1, 'id': 60, 'lvis_id': 136, 'name': 'bow_(weapon)'},
    {'color': [242, 7, 55], 'isthing': 1, 'id': 61, 'lvis_id': 139, 'name': 'bowl'},
    {'color': [23, 166, 98], 'isthing': 1, 'id': 62, 'lvis_id': 146, 'name': 'bracelet'},
    {'color': [15, 119, 207], 'isthing': 1, 'id': 63, 'lvis_id': 152, 'name': 'briefcase'},
    {'color': [177, 173, 83], 'isthing': 1, 'id': 64, 'lvis_id': 156, 'name': 'broom'},
    {'color': [209, 227, 220], 'isthing': 1, 'id': 65, 'lvis_id': 160, 'name': 'bucket'},
    {'color': [243, 245, 218], 'isthing': 1, 'id': 66, 'lvis_id': 162, 'name': 'bull'},
    {'color': [52, 57, 63], 'isthing': 1, 'id': 67, 'lvis_id': 170, 'name': 'bun'},
    {'color': [169, 64, 23], 'isthing': 1, 'id': 68, 'lvis_id': 172, 'name': 'buoy'},
    {'color': [23, 151, 117], 'isthing': 1, 'id': 69, 'lvis_id': 174, 'name': 'bus_(vehicle)'},
    {'color': [233, 145, 138], 'isthing': 1, 'id': 70, 'lvis_id': 175, 'name': 'business_card'},
    {'color': [123, 10, 250], 'isthing': 1, 'id': 71, 'lvis_id': 176, 'name': 'butcher_knife'},
    {'color': [79, 12, 135], 'isthing': 1, 'id': 72, 'lvis_id': 180, 'name': 'cab_(taxi)'},
    {'color': [107, 123, 0], 'isthing': 1, 'id': 73, 'lvis_id': 183, 'name': 'cabinet'},
    {'color': [65, 86, 218], 'isthing': 1, 'id': 74, 'lvis_id': 187, 'name': 'calendar'},
    {'color': [127, 117, 19], 'isthing': 1, 'id': 75, 'lvis_id': 188, 'name': 'calf'},
    {'color': [103, 218, 161], 'isthing': 1, 'id': 76, 'lvis_id': 189, 'name': 'camcorder'},
    {'color': [27, 202, 124], 'isthing': 1, 'id': 77, 'lvis_id': 190, 'name': 'camel'},
    {'color': [7, 109, 246], 'isthing': 1, 'id': 78, 'lvis_id': 191, 'name': 'camera'},
    {'color': [33, 35, 206], 'isthing': 1, 'id': 79, 'lvis_id': 194, 'name': 'can'},
    {'color': [163, 147, 121], 'isthing': 1, 'id': 80, 'lvis_id': 197, 'name': 'candle'},
    {'color': [173, 197, 62], 'isthing': 1, 'id': 81, 'lvis_id': 202, 'name': 'canister'},
    {'color': [209, 96, 190], 'isthing': 1, 'id': 82, 'lvis_id': 204, 'name': 'canoe'},
    {'color': [147, 38, 175], 'isthing': 1, 'id': 83, 'lvis_id': 206, 'name': 'canteen'},
    {'color': [17, 255, 85], 'isthing': 1, 'id': 84, 'lvis_id': 208, 'name': 'bottle_cap'},
    {'color': [96, 21, 128], 'isthing': 1, 'id': 85, 'lvis_id': 209, 'name': 'cape'},
    {'color': [255, 245, 42], 'isthing': 1, 'id': 86, 'lvis_id': 211, 'name': 'car_(automobile)'},
    {'color': [37, 170, 80], 'isthing': 1, 'id': 87, 'lvis_id': 212, 'name': 'railcar_(part_of_a_train)'},
    {'color': [45, 84, 175], 'isthing': 1, 'id': 88, 'lvis_id': 214, 'name': 'car_battery'},
    {'color': [234, 98, 50], 'isthing': 1, 'id': 89, 'lvis_id': 216, 'name': 'card'},
    {'color': [124, 22, 46], 'isthing': 1, 'id': 90, 'lvis_id': 217, 'name': 'cardigan'},
    {'color': [233, 134, 77], 'isthing': 1, 'id': 91, 'lvis_id': 221, 'name': 'carrot'},
    {'color': [26, 2, 83], 'isthing': 1, 'id': 92, 'lvis_id': 222, 'name': 'tote_bag'},
    {'color': [94, 3, 142], 'isthing': 1, 'id': 93, 'lvis_id': 223, 'name': 'cart'},
    {'color': [131, 73, 229], 'isthing': 1, 'id': 94, 'lvis_id': 224, 'name': 'carton'},
    {'color': [206, 248, 141], 'isthing': 1, 'id': 95, 'lvis_id': 229, 'name': 'cat'},
    {'color': [25, 30, 90], 'isthing': 1, 'id': 96, 'lvis_id': 235, 'name': 'cellular_telephone'},
    {'color': [7, 28, 15], 'isthing': 1, 'id': 97, 'lvis_id': 236, 'name': 'chain_mail'},
    {'color': [110, 38, 131], 'isthing': 1, 'id': 98, 'lvis_id': 237, 'name': 'chair'},
    {'color': [214, 161, 142], 'isthing': 1, 'id': 99, 'lvis_id': 247, 'name': 'chicken_(animal)'},
    {'color': [51, 183, 57], 'isthing': 1, 'id': 100, 'lvis_id': 254, 'name': 'crisp_(potato_chip)'},
    {'color': [210, 241, 141], 'isthing': 1, 'id': 101, 'lvis_id': 256, 'name': 'chocolate_bar'},
    {'color': [65, 214, 93], 'isthing': 1, 'id': 102, 'lvis_id': 261, 'name': 'chopping_board'},
    {'color': [118, 121, 239], 'isthing': 1, 'id': 103, 'lvis_id': 262, 'name': 'chopstick'},
    {'color': [235, 194, 114], 'isthing': 1, 'id': 104, 'lvis_id': 266, 'name': 'cigar_box'},
    {'color': [36, 139, 195], 'isthing': 1, 'id': 105, 'lvis_id': 267, 'name': 'cigarette'},
    {'color': [88, 101, 128], 'isthing': 1, 'id': 106, 'lvis_id': 268, 'name': 'cigarette_case'},
    {'color': [179, 130, 115], 'isthing': 1, 'id': 107, 'lvis_id': 274, 'name': 'clip'},
    {'color': [13, 65, 0], 'isthing': 1, 'id': 108, 'lvis_id': 275, 'name': 'clipboard'},
    {'color': [196, 212, 200], 'isthing': 1, 'id': 109, 'lvis_id': 276, 'name': 'clock'},
    {'color': [237, 242, 3], 'isthing': 1, 'id': 110, 'lvis_id': 278, 'name': 'clothes_hamper'},
    {'color': [7, 5, 78], 'isthing': 1, 'id': 111, 'lvis_id': 280, 'name': 'clutch_bag'},
    {'color': [146, 168, 139], 'isthing': 1, 'id': 112, 'lvis_id': 282, 'name': 'coat'},
    {'color': [188, 96, 171], 'isthing': 1, 'id': 113, 'lvis_id': 283, 'name': 'coat_hanger'},
    {'color': [107, 197, 71], 'isthing': 1, 'id': 114, 'lvis_id': 285, 'name': 'cock'},
    {'color': [118, 194, 232], 'isthing': 1, 'id': 115, 'lvis_id': 287, 'name': 'coffee_filter'},
    {'color': [230, 14, 235], 'isthing': 1, 'id': 116, 'lvis_id': 289, 'name': 'coffee_table'},
    {'color': [180, 198, 73], 'isthing': 1, 'id': 117, 'lvis_id': 290, 'name': 'coffeepot'},
    {'color': [157, 133, 133], 'isthing': 1, 'id': 118, 'lvis_id': 292, 'name': 'coin'},
    {'color': [235, 182, 212], 'isthing': 1, 'id': 119, 'lvis_id': 297, 'name': 'pacifier'},
    {'color': [16, 64, 252], 'isthing': 1, 'id': 120, 'lvis_id': 299, 'name': 'computer_keyboard'},
    {'color': [71, 131, 167], 'isthing': 1, 'id': 121, 'lvis_id': 301, 'name': 'cone'},
    {'color': [12, 102, 76], 'isthing': 1, 'id': 122, 'lvis_id': 302, 'name': 'control'},
    {'color': [151, 241, 104], 'isthing': 1, 'id': 123, 'lvis_id': 303, 'name': 'convertible_(automobile)'},
    {'color': [240, 171, 157], 'isthing': 1, 'id': 124, 'lvis_id': 307, 'name': 'cooking_utensil'},
    {'color': [102, 172, 36], 'isthing': 1, 'id': 125, 'lvis_id': 308, 'name': 'cooler_(for_food)'},
    {'color': [132, 177, 234], 'isthing': 1, 'id': 126, 'lvis_id': 312, 'name': 'edible_corn'},
    {'color': [159, 130, 211], 'isthing': 1, 'id': 127, 'lvis_id': 314, 'name': 'cornet'},
    {'color': [135, 215, 228], 'isthing': 1, 'id': 128, 'lvis_id': 323, 'name': 'cowboy_hat'},
    {'color': [62, 14, 244], 'isthing': 1, 'id': 129, 'lvis_id': 324, 'name': 'crab_(animal)'},
    {'color': [142, 236, 126], 'isthing': 1, 'id': 130, 'lvis_id': 325, 'name': 'cracker'},
    {'color': [128, 87, 107], 'isthing': 1, 'id': 131, 'lvis_id': 327, 'name': 'crate'},
    {'color': [222, 203, 202], 'isthing': 1, 'id': 132, 'lvis_id': 336, 'name': 'crow'},
    {'color': [143, 243, 63], 'isthing': 1, 'id': 133, 'lvis_id': 341, 'name': 'crumb'},
    {'color': [15, 234, 176], 'isthing': 1, 'id': 134, 'lvis_id': 342, 'name': 'crutch'},
    {'color': [45, 102, 110], 'isthing': 1, 'id': 135, 'lvis_id': 343, 'name': 'cub_(animal)'},
    {'color': [184, 18, 94], 'isthing': 1, 'id': 136, 'lvis_id': 344, 'name': 'cube'},
    {'color': [62, 162, 228], 'isthing': 1, 'id': 137, 'lvis_id': 345, 'name': 'cucumber'},
    {'color': [125, 57, 138], 'isthing': 1, 'id': 138, 'lvis_id': 347, 'name': 'cup'},
    {'color': [179, 17, 38], 'isthing': 1, 'id': 139, 'lvis_id': 349, 'name': 'cupcake'},
    {'color': [44, 93, 12], 'isthing': 1, 'id': 140, 'lvis_id': 352, 'name': 'curtain'},
    {'color': [25, 157, 122], 'isthing': 1, 'id': 141, 'lvis_id': 353, 'name': 'cushion'},
    {'color': [243, 121, 231], 'isthing': 1, 'id': 142, 'lvis_id': 355, 'name': 'cutting_tool'},
    {'color': [34, 2, 93], 'isthing': 1, 'id': 143, 'lvis_id': 356, 'name': 'cylinder'},
    {'color': [224, 218, 221], 'isthing': 1, 'id': 144, 'lvis_id': 357, 'name': 'cymbal'},
    {'color': [58, 123, 171], 'isthing': 1, 'id': 145, 'lvis_id': 363, 'name': 'deer'},
    {'color': [230, 211, 184], 'isthing': 1, 'id': 146, 'lvis_id': 365, 'name': 'desk'},
    {'color': [130, 43, 82], 'isthing': 1, 'id': 147, 'lvis_id': 369, 'name': 'die'},
    {'color': [41, 134, 229], 'isthing': 1, 'id': 148, 'lvis_id': 371, 'name': 'dining_table'},
    {'color': [85, 243, 38], 'isthing': 1, 'id': 149, 'lvis_id': 373, 'name': 'dish'},
    {'color': [183, 221, 53], 'isthing': 1, 'id': 150, 'lvis_id': 380, 'name': 'dispenser'},
    {'color': [176, 223, 160], 'isthing': 1, 'id': 151, 'lvis_id': 382, 'name': 'dog'},
    {'color': [115, 221, 198], 'isthing': 1, 'id': 152, 'lvis_id': 391, 'name': 'doormat'},
    {'color': [208, 126, 186], 'isthing': 1, 'id': 153, 'lvis_id': 395, 'name': 'drawer'},
    {'color': [217, 90, 73], 'isthing': 1, 'id': 154, 'lvis_id': 398, 'name': 'dress_hat'},
    {'color': [149, 22, 14], 'isthing': 1, 'id': 155, 'lvis_id': 403, 'name': 'drone'},
    {'color': [129, 149, 84], 'isthing': 1, 'id': 156, 'lvis_id': 405, 'name': 'drum_(musical_instrument)'},
    {'color': [4, 56, 134], 'isthing': 1, 'id': 157, 'lvis_id': 406, 'name': 'drumstick'},
    {'color': [8, 125, 162], 'isthing': 1, 'id': 158, 'lvis_id': 407, 'name': 'duck'},
    {'color': [55, 188, 127], 'isthing': 1, 'id': 159, 'lvis_id': 408, 'name': 'duckling'},
    {'color': [174, 118, 254], 'isthing': 1, 'id': 160, 'lvis_id': 410, 'name': 'duffel_bag'},
    {'color': [55, 196, 149], 'isthing': 1, 'id': 161, 'lvis_id': 413, 'name': 'dustpan'},
    {'color': [183, 153, 111], 'isthing': 1, 'id': 162, 'lvis_id': 415, 'name': 'eagle'},
    {'color': [220, 47, 231], 'isthing': 1, 'id': 163, 'lvis_id': 416, 'name': 'earphone'},
    {'color': [39, 94, 50], 'isthing': 1, 'id': 164, 'lvis_id': 418, 'name': 'earring'},
    {'color': [211, 214, 188], 'isthing': 1, 'id': 165, 'lvis_id': 422, 'name': 'egg'},
    {'color': [133, 35, 34], 'isthing': 1, 'id': 166, 'lvis_id': 425, 'name': 'eggbeater'},
    {'color': [203, 197, 155], 'isthing': 1, 'id': 167, 'lvis_id': 428, 'name': 'refrigerator'},
    {'color': [127, 89, 190], 'isthing': 1, 'id': 168, 'lvis_id': 429, 'name': 'elephant'},
    {'color': [132, 84, 37], 'isthing': 1, 'id': 169, 'lvis_id': 431, 'name': 'envelope'},
    {'color': [45, 64, 255], 'isthing': 1, 'id': 170, 'lvis_id': 436, 'name': 'fan'},
    {'color': [176, 115, 201], 'isthing': 1, 'id': 171, 'lvis_id': 437, 'name': 'faucet'},
    {'color': [177, 99, 74], 'isthing': 1, 'id': 172, 'lvis_id': 440, 'name': 'Ferris_wheel'},
    {'color': [77, 74, 253], 'isthing': 1, 'id': 173, 'lvis_id': 446, 'name': 'file_(tool)'},
    {'color': [119, 89, 113], 'isthing': 1, 'id': 174, 'lvis_id': 448, 'name': 'fire_engine'},
    {'color': [49, 48, 84], 'isthing': 1, 'id': 175, 'lvis_id': 453, 'name': 'fish'},
    {'color': [180, 33, 221], 'isthing': 1, 'id': 176, 'lvis_id': 457, 'name': 'fishing_rod'},
    {'color': [74, 0, 105], 'isthing': 1, 'id': 177, 'lvis_id': 458, 'name': 'flag'},
    {'color': [214, 1, 60], 'isthing': 1, 'id': 178, 'lvis_id': 463, 'name': 'flashlight'},
    {'color': [148, 63, 17], 'isthing': 1, 'id': 179, 'lvis_id': 468, 'name': 'flute_glass'},
    {'color': [13, 148, 129], 'isthing': 1, 'id': 180, 'lvis_id': 473, 'name': 'football_helmet'},
    {'color': [51, 76, 170], 'isthing': 1, 'id': 181, 'lvis_id': 474, 'name': 'footstool'},
    {'color': [137, 206, 109], 'isthing': 1, 'id': 182, 'lvis_id': 475, 'name': 'fork'},
    {'color': [1, 82, 233], 'isthing': 1, 'id': 183, 'lvis_id': 480, 'name': 'frisbee'},
    {'color': [46, 110, 107], 'isthing': 1, 'id': 184, 'lvis_id': 481, 'name': 'frog'},
    {'color': [223, 1, 32], 'isthing': 1, 'id': 185, 'lvis_id': 484, 'name': 'frying_pan'},
    {'color': [41, 106, 252], 'isthing': 1, 'id': 186, 'lvis_id': 489, 'name': 'garbage'},
    {'color': [130, 111, 139], 'isthing': 1, 'id': 187, 'lvis_id': 490, 'name': 'garbage_truck'},
    {'color': [201, 216, 223], 'isthing': 1, 'id': 188, 'lvis_id': 491, 'name': 'garden_hose'},
    {'color': [133, 146, 207], 'isthing': 1, 'id': 189, 'lvis_id': 495, 'name': 'gasmask'},
    {'color': [57, 19, 22], 'isthing': 1, 'id': 190, 'lvis_id': 499, 'name': 'giant_panda'},
    {'color': [207, 155, 78], 'isthing': 1, 'id': 191, 'lvis_id': 500, 'name': 'gift_wrap'},
    {'color': [213, 105, 134], 'isthing': 1, 'id': 192, 'lvis_id': 502, 'name': 'giraffe'},
    {'color': [93, 239, 147], 'isthing': 1, 'id': 193, 'lvis_id': 506, 'name': 'glove'},
    {'color': [252, 104, 234], 'isthing': 1, 'id': 194, 'lvis_id': 507, 'name': 'goat'},
    {'color': [215, 78, 75], 'isthing': 1, 'id': 195, 'lvis_id': 508, 'name': 'goggles'},
    {'color': [241, 27, 45], 'isthing': 1, 'id': 196, 'lvis_id': 510, 'name': 'golf_club'},
    {'color': [19, 31, 90], 'isthing': 1, 'id': 197, 'lvis_id': 511, 'name': 'golfcart'},
    {'color': [196, 212, 235], 'isthing': 1, 'id': 198, 'lvis_id': 513, 'name': 'goose'},
    {'color': [176, 83, 90], 'isthing': 1, 'id': 199, 'lvis_id': 514, 'name': 'gorilla'},
    {'color': [155, 165, 68], 'isthing': 1, 'id': 200, 'lvis_id': 529, 'name': 'grocery_bag'},
    {'color': [212, 161, 109], 'isthing': 1, 'id': 201, 'lvis_id': 531, 'name': 'guitar'},
    {'color': [251, 101, 212], 'isthing': 1, 'id': 202, 'lvis_id': 533, 'name': 'gun'},
    {'color': [35, 185, 213], 'isthing': 1, 'id': 203, 'lvis_id': 534, 'name': 'hair_spray'},
    {'color': [48, 53, 128], 'isthing': 1, 'id': 204, 'lvis_id': 535, 'name': 'hairbrush'},
    {'color': [95, 188, 38], 'isthing': 1, 'id': 205, 'lvis_id': 539, 'name': 'hamburger'},
    {'color': [73, 222, 220], 'isthing': 1, 'id': 206, 'lvis_id': 540, 'name': 'hammer'},
    {'color': [5, 104, 174], 'isthing': 1, 'id': 207, 'lvis_id': 543, 'name': 'hamster'},
    {'color': [123, 67, 197], 'isthing': 1, 'id': 208, 'lvis_id': 544, 'name': 'hair_dryer'},
    {'color': [235, 196, 33], 'isthing': 1, 'id': 209, 'lvis_id': 546, 'name': 'hand_towel'},
    {'color': [136, 172, 211], 'isthing': 1, 'id': 210, 'lvis_id': 547, 'name': 'handcart'},
    {'color': [131, 23, 89], 'isthing': 1, 'id': 211, 'lvis_id': 548, 'name': 'handcuff'},
    {'color': [79, 13, 225], 'isthing': 1, 'id': 212, 'lvis_id': 549, 'name': 'handkerchief'},
    {'color': [86, 231, 247], 'isthing': 1, 'id': 213, 'lvis_id': 550, 'name': 'handle'},
    {'color': [94, 91, 199], 'isthing': 1, 'id': 214, 'lvis_id': 551, 'name': 'handsaw'},
    {'color': [120, 124, 225], 'isthing': 1, 'id': 215, 'lvis_id': 553, 'name': 'harmonium'},
    {'color': [143, 224, 55], 'isthing': 1, 'id': 216, 'lvis_id': 554, 'name': 'hat'},
    {'color': [134, 60, 83], 'isthing': 1, 'id': 217, 'lvis_id': 561, 'name': 'headscarf'},
    {'color': [56, 198, 177], 'isthing': 1, 'id': 218, 'lvis_id': 567, 'name': 'helicopter'},
    {'color': [29, 253, 105], 'isthing': 1, 'id': 219, 'lvis_id': 568, 'name': 'helmet'},
    {'color': [181, 114, 84], 'isthing': 1, 'id': 220, 'lvis_id': 569, 'name': 'heron'},
    {'color': [137, 129, 102], 'isthing': 1, 'id': 221, 'lvis_id': 572, 'name': 'hippopotamus'},
    {'color': [200, 57, 114], 'isthing': 1, 'id': 222, 'lvis_id': 573, 'name': 'hockey_stick'},
    {'color': [185, 59, 36], 'isthing': 1, 'id': 223, 'lvis_id': 574, 'name': 'hog'},
    {'color': [133, 198, 48], 'isthing': 1, 'id': 224, 'lvis_id': 579, 'name': 'horse'},
    {'color': [19, 122, 68], 'isthing': 1, 'id': 225, 'lvis_id': 580, 'name': 'hose'},
    {'color': [240, 72, 240], 'isthing': 1, 'id': 226, 'lvis_id': 588, 'name': 'polar_bear'},
    {'color': [155, 63, 185], 'isthing': 1, 'id': 227, 'lvis_id': 589, 'name': 'icecream'},
    {'color': [44, 145, 93], 'isthing': 1, 'id': 228, 'lvis_id': 595, 'name': 'igniter'},
    {'color': [140, 136, 113], 'isthing': 1, 'id': 229, 'lvis_id': 598, 'name': 'iPod'},
    {'color': [93, 100, 95], 'isthing': 1, 'id': 230, 'lvis_id': 599, 'name': 'iron_(for_clothing)'},
    {'color': [17, 191, 13], 'isthing': 1, 'id': 231, 'lvis_id': 601, 'name': 'jacket'},
    {'color': [238, 54, 233], 'isthing': 1, 'id': 232, 'lvis_id': 603, 'name': 'jean'},
    {'color': [135, 167, 173], 'isthing': 1, 'id': 233, 'lvis_id': 604, 'name': 'jeep'},
    {'color': [254, 158, 177], 'isthing': 1, 'id': 234, 'lvis_id': 606, 'name': 'jersey'},
    {'color': [209, 32, 56], 'isthing': 1, 'id': 235, 'lvis_id': 611, 'name': 'kayak'},
    {'color': [139, 110, 17], 'isthing': 1, 'id': 236, 'lvis_id': 614, 'name': 'kettle'},
    {'color': [156, 153, 44], 'isthing': 1, 'id': 237, 'lvis_id': 621, 'name': 'kite'},
    {'color': [211, 146, 153], 'isthing': 1, 'id': 238, 'lvis_id': 625, 'name': 'knife'},
    {'color': [135, 1, 81], 'isthing': 1, 'id': 239, 'lvis_id': 627, 'name': 'knitting_needle'},
    {'color': [113, 108, 178], 'isthing': 1, 'id': 240, 'lvis_id': 628, 'name': 'knob'},
    {'color': [174, 253, 21], 'isthing': 1, 'id': 241, 'lvis_id': 633, 'name': 'ladle'},
    {'color': [253, 251, 83], 'isthing': 1, 'id': 242, 'lvis_id': 637, 'name': 'lamp'},
    {'color': [172, 50, 1], 'isthing': 1, 'id': 243, 'lvis_id': 641, 'name': 'lanyard'},
    {'color': [148, 236, 70], 'isthing': 1, 'id': 244, 'lvis_id': 642, 'name': 'laptop_computer'},
    {'color': [45, 104, 99], 'isthing': 1, 'id': 245, 'lvis_id': 644, 'name': 'latch'},
    {'color': [32, 20, 132], 'isthing': 1, 'id': 246, 'lvis_id': 645, 'name': 'lawn_mower'},
    {'color': [223, 67, 144], 'isthing': 1, 'id': 247, 'lvis_id': 649, 'name': 'lemon'},
    {'color': [247, 163, 199], 'isthing': 1, 'id': 248, 'lvis_id': 651, 'name': 'lettuce'},
    {'color': [80, 236, 97], 'isthing': 1, 'id': 249, 'lvis_id': 653, 'name': 'life_buoy'},
    {'color': [74, 221, 26], 'isthing': 1, 'id': 250, 'lvis_id': 654, 'name': 'life_jacket'},
    {'color': [234, 220, 16], 'isthing': 1, 'id': 251, 'lvis_id': 660, 'name': 'lion'},
    {'color': [60, 27, 0], 'isthing': 1, 'id': 252, 'lvis_id': 664, 'name': 'lizard'},
    {'color': [26, 13, 73], 'isthing': 1, 'id': 253, 'lvis_id': 666, 'name': 'log'},
    {'color': [153, 211, 0], 'isthing': 1, 'id': 254, 'lvis_id': 672, 'name': 'magazine'},
    {'color': [243, 70, 173], 'isthing': 1, 'id': 255, 'lvis_id': 675, 'name': 'mailbox_(at_home)'},
    {'color': [129, 17, 98], 'isthing': 1, 'id': 256, 'lvis_id': 676, 'name': 'mallet'},
    {'color': [187, 38, 214], 'isthing': 1, 'id': 257, 'lvis_id': 682, 'name': 'marker'},
    {'color': [170, 13, 137], 'isthing': 1, 'id': 258, 'lvis_id': 689, 'name': 'mat_(gym_equipment)'},
    {'color': [57, 120, 31], 'isthing': 1, 'id': 259, 'lvis_id': 690, 'name': 'matchbox'},
    {'color': [18, 138, 24], 'isthing': 1, 'id': 260, 'lvis_id': 691, 'name': 'mattress'},
    {'color': [61, 223, 154], 'isthing': 1, 'id': 261, 'lvis_id': 692, 'name': 'measuring_cup'},
    {'color': [240, 148, 21], 'isthing': 1, 'id': 262, 'lvis_id': 693, 'name': 'measuring_stick'},
    {'color': [152, 97, 118], 'isthing': 1, 'id': 263, 'lvis_id': 695, 'name': 'medicine'},
    {'color': [161, 198, 171], 'isthing': 1, 'id': 264, 'lvis_id': 697, 'name': 'microphone'},
    {'color': [189, 124, 183], 'isthing': 1, 'id': 265, 'lvis_id': 702, 'name': 'minivan'},
    {'color': [77, 196, 202], 'isthing': 1, 'id': 266, 'lvis_id': 704, 'name': 'mirror'},
    {'color': [98, 248, 15], 'isthing': 1, 'id': 267, 'lvis_id': 706, 'name': 'mixer_(kitchen_tool)'},
    {'color': [61, 167, 150], 'isthing': 1, 'id': 268, 'lvis_id': 707, 'name': 'money'},
    {'color': [96, 141, 108], 'isthing': 1, 'id': 269, 'lvis_id': 708, 'name': 'monitor_(computer_equipment) computer_monitor'},
    {'color': [165, 235, 13], 'isthing': 1, 'id': 270, 'lvis_id': 709, 'name': 'monkey'},
    {'color': [38, 116, 86], 'isthing': 1, 'id': 271, 'lvis_id': 711, 'name': 'motor_scooter'},
    {'color': [115, 210, 146], 'isthing': 1, 'id': 272, 'lvis_id': 713, 'name': 'motorboat'},
    {'color': [234, 162, 182], 'isthing': 1, 'id': 273, 'lvis_id': 714, 'name': 'motorcycle'},
    {'color': [145, 203, 235], 'isthing': 1, 'id': 274, 'lvis_id': 716, 'name': 'mouse_(animal_rodent)'},
    {'color': [152, 17, 172], 'isthing': 1, 'id': 275, 'lvis_id': 717, 'name': 'mouse_(computer_equipment)'},
    {'color': [117, 74, 138], 'isthing': 1, 'id': 276, 'lvis_id': 723, 'name': 'musical_instrument'},
    {'color': [238, 117, 213], 'isthing': 1, 'id': 277, 'lvis_id': 726, 'name': 'napkin'},
    {'color': [244, 247, 244], 'isthing': 1, 'id': 278, 'lvis_id': 728, 'name': 'necklace'},
    {'color': [49, 24, 177], 'isthing': 1, 'id': 279, 'lvis_id': 731, 'name': 'nest'},
    {'color': [157, 100, 208], 'isthing': 1, 'id': 280, 'lvis_id': 732, 'name': 'newsstand'},
    {'color': [71, 137, 16], 'isthing': 1, 'id': 281, 'lvis_id': 736, 'name': 'notebook'},
    {'color': [10, 198, 65], 'isthing': 1, 'id': 282, 'lvis_id': 737, 'name': 'notepad'},
    {'color': [149, 84, 189], 'isthing': 1, 'id': 283, 'lvis_id': 738, 'name': 'nut'},
    {'color': [185, 89, 39], 'isthing': 1, 'id': 284, 'lvis_id': 740, 'name': 'oar'},
    {'color': [143, 85, 207], 'isthing': 1, 'id': 285, 'lvis_id': 746, 'name': 'onion'},
    {'color': [209, 77, 0], 'isthing': 1, 'id': 286, 'lvis_id': 747, 'name': 'orange_(fruit)'},
    {'color': [73, 220, 47], 'isthing': 1, 'id': 287, 'lvis_id': 754, 'name': 'packet'},
    {'color': [3, 190, 186], 'isthing': 1, 'id': 288, 'lvis_id': 757, 'name': 'paddle'},
    {'color': [239, 113, 233], 'isthing': 1, 'id': 289, 'lvis_id': 759, 'name': 'paintbox'},
    {'color': [97, 138, 106], 'isthing': 1, 'id': 290, 'lvis_id': 760, 'name': 'paintbrush'},
    {'color': [189, 223, 180], 'isthing': 1, 'id': 291, 'lvis_id': 761, 'name': 'painting'},
    {'color': [197, 4, 5], 'isthing': 1, 'id': 292, 'lvis_id': 762, 'name': 'pajamas'},
    {'color': [187, 241, 190], 'isthing': 1, 'id': 293, 'lvis_id': 763, 'name': 'palette'},
    {'color': [159, 178, 112], 'isthing': 1, 'id': 294, 'lvis_id': 764, 'name': 'pan_(for_cooking)'},
    {'color': [82, 31, 82], 'isthing': 1, 'id': 295, 'lvis_id': 771, 'name': 'paper_towel'},
    {'color': [156, 230, 26], 'isthing': 1, 'id': 296, 'lvis_id': 774, 'name': 'parachute'},
    {'color': [161, 42, 71], 'isthing': 1, 'id': 297, 'lvis_id': 780, 'name': 'parrot'},
    {'color': [14, 133, 16], 'isthing': 1, 'id': 298, 'lvis_id': 790, 'name': 'peeler_(tool_for_fruit_and_vegetables)'},
    {'color': [193, 230, 87], 'isthing': 1, 'id': 299, 'lvis_id': 792, 'name': 'pelican'},
    {'color': [140, 62, 36], 'isthing': 1, 'id': 300, 'lvis_id': 793, 'name': 'pen'},
    {'color': [113, 107, 161], 'isthing': 1, 'id': 301, 'lvis_id': 794, 'name': 'pencil'},
    {'color': [41, 198, 56], 'isthing': 1, 'id': 302, 'lvis_id': 798, 'name': 'penguin'},
    {'color': [63, 85, 139], 'isthing': 1, 'id': 303, 'lvis_id': 802, 'name': 'pepper_mill'},
    {'color': [196, 12, 49], 'isthing': 1, 'id': 304, 'lvis_id': 804, 'name': 'persimmon'},
    {'color': [251, 13, 143], 'isthing': 1, 'id': 305, 'lvis_id': 805, 'name': 'person'},
    {'color': [161, 59, 153], 'isthing': 1, 'id': 306, 'lvis_id': 807, 'name': 'petfood'},
    {'color': [249, 161, 112], 'isthing': 1, 'id': 307, 'lvis_id': 810, 'name': 'phonograph_record'},
    {'color': [142, 36, 120], 'isthing': 1, 'id': 308, 'lvis_id': 811, 'name': 'piano'},
    {'color': [240, 250, 129], 'isthing': 1, 'id': 309, 'lvis_id': 812, 'name': 'pickle'},
    {'color': [255, 180, 226], 'isthing': 1, 'id': 310, 'lvis_id': 813, 'name': 'pickup_truck'},
    {'color': [239, 115, 148], 'isthing': 1, 'id': 311, 'lvis_id': 815, 'name': 'pigeon'},
    {'color': [207, 191, 90], 'isthing': 1, 'id': 312, 'lvis_id': 817, 'name': 'pillow'},
    {'color': [61, 212, 76], 'isthing': 1, 'id': 313, 'lvis_id': 821, 'name': 'ping-pong_ball'},
    {'color': [110, 238, 81], 'isthing': 1, 'id': 314, 'lvis_id': 823, 'name': 'tobacco_pipe'},
    {'color': [129, 27, 53], 'isthing': 1, 'id': 315, 'lvis_id': 824, 'name': 'pipe'},
    {'color': [25, 83, 25], 'isthing': 1, 'id': 316, 'lvis_id': 825, 'name': 'pistol'},
    {'color': [109, 20, 110], 'isthing': 1, 'id': 317, 'lvis_id': 827, 'name': 'pitcher_(vessel_for_liquid)'},
    {'color': [21, 207, 2], 'isthing': 1, 'id': 318, 'lvis_id': 829, 'name': 'pizza'},
    {'color': [88, 80, 188], 'isthing': 1, 'id': 319, 'lvis_id': 831, 'name': 'plate'},
    {'color': [74, 131, 158], 'isthing': 1, 'id': 320, 'lvis_id': 835, 'name': 'pliers'},
    {'color': [51, 196, 253], 'isthing': 1, 'id': 321, 'lvis_id': 840, 'name': 'pole'},
    {'color': [167, 34, 222], 'isthing': 1, 'id': 322, 'lvis_id': 844, 'name': 'pony'},
    {'color': [172, 175, 107], 'isthing': 1, 'id': 323, 'lvis_id': 850, 'name': 'poster'},
    {'color': [246, 185, 174], 'isthing': 1, 'id': 324, 'lvis_id': 851, 'name': 'pot'},
    {'color': [118, 239, 73], 'isthing': 1, 'id': 325, 'lvis_id': 852, 'name': 'flowerpot'},
    {'color': [117, 201, 159], 'isthing': 1, 'id': 326, 'lvis_id': 853, 'name': 'potato'},
    {'color': [235, 118, 63], 'isthing': 1, 'id': 327, 'lvis_id': 856, 'name': 'pouch'},
    {'color': [57, 54, 165], 'isthing': 1, 'id': 328, 'lvis_id': 857, 'name': 'power_shovel'},
    {'color': [36, 225, 177], 'isthing': 1, 'id': 329, 'lvis_id': 868, 'name': 'pumpkin'},
    {'color': [212, 27, 230], 'isthing': 1, 'id': 330, 'lvis_id': 870, 'name': 'puppet'},
    {'color': [212, 227, 191], 'isthing': 1, 'id': 331, 'lvis_id': 875, 'name': 'rabbit'},
    {'color': [186, 37, 30], 'isthing': 1, 'id': 332, 'lvis_id': 876, 'name': 'race_car'},
    {'color': [245, 201, 226], 'isthing': 1, 'id': 333, 'lvis_id': 877, 'name': 'racket'},
    {'color': [247, 112, 199], 'isthing': 1, 'id': 334, 'lvis_id': 882, 'name': 'raft'},
    {'color': [141, 209, 90], 'isthing': 1, 'id': 335, 'lvis_id': 883, 'name': 'rag_doll'},
    {'color': [213, 193, 217], 'isthing': 1, 'id': 336, 'lvis_id': 887, 'name': 'rat'},
    {'color': [107, 155, 33], 'isthing': 1, 'id': 337, 'lvis_id': 888, 'name': 'razorblade'},
    {'color': [154, 8, 4], 'isthing': 1, 'id': 338, 'lvis_id': 893, 'name': 'record_player'},
    {'color': [154, 241, 29], 'isthing': 1, 'id': 339, 'lvis_id': 896, 'name': 'remote_control'},
    {'color': [74, 16, 33], 'isthing': 1, 'id': 340, 'lvis_id': 897, 'name': 'rhinoceros'},
    {'color': [2, 151, 246], 'isthing': 1, 'id': 341, 'lvis_id': 899, 'name': 'rifle'},
    {'color': [148, 217, 53], 'isthing': 1, 'id': 342, 'lvis_id': 910, 'name': 'rubber_band'},
    {'color': [241, 78, 37], 'isthing': 1, 'id': 343, 'lvis_id': 911, 'name': 'runner_(carpet)'},
    {'color': [107, 76, 226], 'isthing': 1, 'id': 344, 'lvis_id': 914, 'name': 'saddle_blanket'},
    {'color': [103, 41, 216], 'isthing': 1, 'id': 345, 'lvis_id': 924, 'name': 'saltshaker'},
    {'color': [133, 126, 58], 'isthing': 1, 'id': 346, 'lvis_id': 925, 'name': 'sandal_(type_of_shoe)'},
    {'color': [145, 207, 47], 'isthing': 1, 'id': 347, 'lvis_id': 926, 'name': 'sandwich'},
    {'color': [76, 242, 198], 'isthing': 1, 'id': 348, 'lvis_id': 928, 'name': 'saucepan'},
    {'color': [132, 66, 251], 'isthing': 1, 'id': 349, 'lvis_id': 932, 'name': 'saxophone'},
    {'color': [141, 205, 32], 'isthing': 1, 'id': 350, 'lvis_id': 935, 'name': 'scarf'},
    {'color': [168, 210, 150], 'isthing': 1, 'id': 351, 'lvis_id': 936, 'name': 'school_bus'},
    {'color': [13, 195, 38], 'isthing': 1, 'id': 352, 'lvis_id': 937, 'name': 'scissors'},
    {'color': [143, 120, 231], 'isthing': 1, 'id': 353, 'lvis_id': 938, 'name': 'scoreboard'},
    {'color': [234, 152, 201], 'isthing': 1, 'id': 354, 'lvis_id': 940, 'name': 'scraper'},
    {'color': [70, 210, 104], 'isthing': 1, 'id': 355, 'lvis_id': 942, 'name': 'screwdriver'},
    {'color': [31, 144, 18], 'isthing': 1, 'id': 356, 'lvis_id': 943, 'name': 'scrubbing_brush'},
    {'color': [54, 174, 190], 'isthing': 1, 'id': 357, 'lvis_id': 944, 'name': 'sculpture'},
    {'color': [18, 122, 242], 'isthing': 1, 'id': 358, 'lvis_id': 950, 'name': 'serving_dish'},
    {'color': [155, 179, 188], 'isthing': 1, 'id': 359, 'lvis_id': 952, 'name': 'shaker'},
    {'color': [21, 10, 27], 'isthing': 1, 'id': 360, 'lvis_id': 953, 'name': 'shampoo'},
    {'color': [48, 12, 170], 'isthing': 1, 'id': 361, 'lvis_id': 954, 'name': 'shark'},
    {'color': [70, 74, 138], 'isthing': 1, 'id': 362, 'lvis_id': 955, 'name': 'sharpener'},
    {'color': [60, 89, 176], 'isthing': 1, 'id': 363, 'lvis_id': 957, 'name': 'shaver_(electric)'},
    {'color': [189, 100, 82], 'isthing': 1, 'id': 364, 'lvis_id': 959, 'name': 'shawl'},
    {'color': [130, 47, 156], 'isthing': 1, 'id': 365, 'lvis_id': 960, 'name': 'shears'},
    {'color': [72, 80, 212], 'isthing': 1, 'id': 366, 'lvis_id': 961, 'name': 'sheep'},
    {'color': [198, 31, 46], 'isthing': 1, 'id': 367, 'lvis_id': 964, 'name': 'shield'},
    {'color': [252, 68, 175], 'isthing': 1, 'id': 368, 'lvis_id': 965, 'name': 'shirt'},
    {'color': [101, 201, 150], 'isthing': 1, 'id': 369, 'lvis_id': 966, 'name': 'shoe'},
    {'color': [106, 131, 54], 'isthing': 1, 'id': 370, 'lvis_id': 968, 'name': 'shopping_cart'},
    {'color': [158, 93, 158], 'isthing': 1, 'id': 371, 'lvis_id': 969, 'name': 'short_pants'},
    {'color': [77, 171, 206], 'isthing': 1, 'id': 372, 'lvis_id': 971, 'name': 'shoulder_bag'},
    {'color': [34, 59, 164], 'isthing': 1, 'id': 373, 'lvis_id': 972, 'name': 'shovel'},
    {'color': [141, 38, 117], 'isthing': 1, 'id': 374, 'lvis_id': 976, 'name': 'sieve'},
    {'color': [28, 154, 121], 'isthing': 1, 'id': 375, 'lvis_id': 980, 'name': 'skateboard'},
    {'color': [44, 246, 37], 'isthing': 1, 'id': 376, 'lvis_id': 982, 'name': 'ski'},
    {'color': [55, 205, 128], 'isthing': 1, 'id': 377, 'lvis_id': 985, 'name': 'ski_pole'},
    {'color': [119, 103, 250], 'isthing': 1, 'id': 378, 'lvis_id': 986, 'name': 'skirt'},
    {'color': [93, 62, 49], 'isthing': 1, 'id': 379, 'lvis_id': 987, 'name': 'sled'},
    {'color': [47, 192, 204], 'isthing': 1, 'id': 380, 'lvis_id': 990, 'name': 'slipper_(footwear)'},
    {'color': [10, 161, 45], 'isthing': 1, 'id': 381, 'lvis_id': 996, 'name': 'soap'},
    {'color': [9, 126, 174], 'isthing': 1, 'id': 382, 'lvis_id': 998, 'name': 'sock'},
    {'color': [126, 166, 74], 'isthing': 1, 'id': 383, 'lvis_id': 1001, 'name': 'sofa'},
    {'color': [219, 30, 221], 'isthing': 1, 'id': 384, 'lvis_id': 1007, 'name': 'soupspoon'},
    {'color': [50, 139, 173], 'isthing': 1, 'id': 385, 'lvis_id': 1012, 'name': 'spatula'},
    {'color': [254, 150, 173], 'isthing': 1, 'id': 386, 'lvis_id': 1014, 'name': 'spectacles'},
    {'color': [191, 209, 192], 'isthing': 1, 'id': 387, 'lvis_id': 1016, 'name': 'spider'},
    {'color': [92, 41, 110], 'isthing': 1, 'id': 388, 'lvis_id': 1017, 'name': 'sponge'},
    {'color': [86, 21, 228], 'isthing': 1, 'id': 389, 'lvis_id': 1018, 'name': 'spoon'},
    {'color': [0, 154, 191], 'isthing': 1, 'id': 390, 'lvis_id': 1020, 'name': 'spotlight'},
    {'color': [73, 71, 189], 'isthing': 1, 'id': 391, 'lvis_id': 1021, 'name': 'squirrel'},
    {'color': [110, 99, 16], 'isthing': 1, 'id': 392, 'lvis_id': 1024, 'name': 'statue_(sculpture)'},
    {'color': [14, 101, 134], 'isthing': 1, 'id': 393, 'lvis_id': 1028, 'name': 'steering_wheel'},
    {'color': [174, 179, 208], 'isthing': 1, 'id': 394, 'lvis_id': 1031, 'name': 'step_stool'},
    {'color': [42, 165, 55], 'isthing': 1, 'id': 395, 'lvis_id': 1037, 'name': 'stool'},
    {'color': [233, 193, 55], 'isthing': 1, 'id': 396, 'lvis_id': 1040, 'name': 'stove'},
    {'color': [118, 51, 233], 'isthing': 1, 'id': 397, 'lvis_id': 1043, 'name': 'straw_(for_drinking)'},
    {'color': [43, 90, 209], 'isthing': 1, 'id': 398, 'lvis_id': 1044, 'name': 'strawberry'},
    {'color': [23, 77, 204], 'isthing': 1, 'id': 399, 'lvis_id': 1045, 'name': 'street_sign'},
    {'color': [4, 238, 235], 'isthing': 1, 'id': 400, 'lvis_id': 1048, 'name': 'stylus'},
    {'color': [208, 157, 38], 'isthing': 1, 'id': 401, 'lvis_id': 1050, 'name': 'sugar_bowl'},
    {'color': [180, 40, 203], 'isthing': 1, 'id': 402, 'lvis_id': 1054, 'name': 'sunglasses'},
    {'color': [171, 18, 244], 'isthing': 1, 'id': 403, 'lvis_id': 1056, 'name': 'sunscreen'},
    {'color': [94, 113, 131], 'isthing': 1, 'id': 404, 'lvis_id': 1057, 'name': 'surfboard'},
    {'color': [28, 251, 52], 'isthing': 1, 'id': 405, 'lvis_id': 1059, 'name': 'mop'},
    {'color': [16, 153, 246], 'isthing': 1, 'id': 406, 'lvis_id': 1060, 'name': 'sweat_pants'},
    {'color': [46, 17, 209], 'isthing': 1, 'id': 407, 'lvis_id': 1062, 'name': 'sweater'},
    {'color': [81, 15, 131], 'isthing': 1, 'id': 408, 'lvis_id': 1063, 'name': 'sweatshirt'},
    {'color': [162, 11, 48], 'isthing': 1, 'id': 409, 'lvis_id': 1066, 'name': 'sword'},
    {'color': [62, 64, 181], 'isthing': 1, 'id': 410, 'lvis_id': 1070, 'name': 'table'},
    {'color': [13, 164, 233], 'isthing': 1, 'id': 411, 'lvis_id': 1071, 'name': 'table_lamp'},
    {'color': [196, 71, 156], 'isthing': 1, 'id': 412, 'lvis_id': 1072, 'name': 'tablecloth'},
    {'color': [24, 153, 112], 'isthing': 1, 'id': 413, 'lvis_id': 1075, 'name': 'tag'},
    {'color': [230, 241, 189], 'isthing': 1, 'id': 414, 'lvis_id': 1078, 'name': 'army_tank'},
    {'color': [158, 56, 169], 'isthing': 1, 'id': 415, 'lvis_id': 1080, 'name': 'tank_top_(clothing)'},
    {'color': [124, 216, 214], 'isthing': 1, 'id': 416, 'lvis_id': 1081, 'name': 'tape_(sticky_cloth_or_paper)'},
    {'color': [201, 151, 66], 'isthing': 1, 'id': 417, 'lvis_id': 1082, 'name': 'tape_measure'},
    {'color': [51, 67, 195], 'isthing': 1, 'id': 418, 'lvis_id': 1084, 'name': 'tarp'},
    {'color': [222, 61, 239], 'isthing': 1, 'id': 419, 'lvis_id': 1088, 'name': 'teacup'},
    {'color': [198, 20, 10], 'isthing': 1, 'id': 420, 'lvis_id': 1089, 'name': 'teakettle'},
    {'color': [69, 236, 253], 'isthing': 1, 'id': 421, 'lvis_id': 1090, 'name': 'teapot'},
    {'color': [175, 254, 211], 'isthing': 1, 'id': 422, 'lvis_id': 1092, 'name': 'telephone'},
    {'color': [209, 241, 175], 'isthing': 1, 'id': 423, 'lvis_id': 1097, 'name': 'television_set'},
    {'color': [14, 13, 43], 'isthing': 1, 'id': 424, 'lvis_id': 1099, 'name': 'tennis_racket'},
    {'color': [41, 235, 177], 'isthing': 1, 'id': 425, 'lvis_id': 1101, 'name': 'thermometer'},
    {'color': [236, 66, 31], 'isthing': 1, 'id': 426, 'lvis_id': 1102, 'name': 'thermos_bottle'},
    {'color': [20, 109, 168], 'isthing': 1, 'id': 427, 'lvis_id': 1105, 'name': 'thread'},
    {'color': [148, 79, 124], 'isthing': 1, 'id': 428, 'lvis_id': 1108, 'name': 'tiger'},
    {'color': [139, 91, 67], 'isthing': 1, 'id': 429, 'lvis_id': 1111, 'name': 'tinfoil'},
    {'color': [64, 222, 58], 'isthing': 1, 'id': 430, 'lvis_id': 1113, 'name': 'tissue_paper'},
    {'color': [86, 95, 173], 'isthing': 1, 'id': 431, 'lvis_id': 1114, 'name': 'toast_(food)'},
    {'color': [5, 245, 235], 'isthing': 1, 'id': 432, 'lvis_id': 1115, 'name': 'toaster'},
    {'color': [227, 115, 192], 'isthing': 1, 'id': 433, 'lvis_id': 1120, 'name': 'tongs'},
    {'color': [137, 199, 174], 'isthing': 1, 'id': 434, 'lvis_id': 1121, 'name': 'toolbox'},
    {'color': [49, 148, 197], 'isthing': 1, 'id': 435, 'lvis_id': 1122, 'name': 'toothbrush'},
    {'color': [232, 127, 58], 'isthing': 1, 'id': 436, 'lvis_id': 1123, 'name': 'toothpaste'},
    {'color': [43, 87, 137], 'isthing': 1, 'id': 437, 'lvis_id': 1124, 'name': 'toothpick'},
    {'color': [231, 236, 22], 'isthing': 1, 'id': 438, 'lvis_id': 1125, 'name': 'cover'},
    {'color': [66, 240, 130], 'isthing': 1, 'id': 439, 'lvis_id': 1127, 'name': 'tow_truck'},
    {'color': [183, 34, 137], 'isthing': 1, 'id': 440, 'lvis_id': 1128, 'name': 'towel'},
    {'color': [67, 81, 43], 'isthing': 1, 'id': 441, 'lvis_id': 1130, 'name': 'toy'},
    {'color': [137, 108, 236], 'isthing': 1, 'id': 442, 'lvis_id': 1131, 'name': 'tractor_(farm_equipment)'},
    {'color': [146, 199, 128], 'isthing': 1, 'id': 443, 'lvis_id': 1132, 'name': 'traffic_light'},
    {'color': [14, 43, 238], 'isthing': 1, 'id': 444, 'lvis_id': 1133, 'name': 'dirt_bike'},
    {'color': [223, 121, 207], 'isthing': 1, 'id': 445, 'lvis_id': 1134, 'name': 'trailer_truck'},
    {'color': [86, 133, 86], 'isthing': 1, 'id': 446, 'lvis_id': 1135, 'name': 'train_(railroad_vehicle)'},
    {'color': [103, 167, 59], 'isthing': 1, 'id': 447, 'lvis_id': 1137, 'name': 'tray'},
    {'color': [190, 11, 200], 'isthing': 1, 'id': 448, 'lvis_id': 1142, 'name': 'tripod'},
    {'color': [231, 109, 46], 'isthing': 1, 'id': 449, 'lvis_id': 1143, 'name': 'trousers'},
    {'color': [151, 118, 111], 'isthing': 1, 'id': 450, 'lvis_id': 1144, 'name': 'truck'},
    {'color': [213, 95, 204], 'isthing': 1, 'id': 451, 'lvis_id': 1149, 'name': 'turkey_(bird)'},
    {'color': [154, 37, 137], 'isthing': 1, 'id': 452, 'lvis_id': 1152, 'name': 'turtle'},
    {'color': [3, 216, 211], 'isthing': 1, 'id': 453, 'lvis_id': 1155, 'name': 'umbrella'},
    {'color': [19, 230, 92], 'isthing': 1, 'id': 454, 'lvis_id': 1156, 'name': 'underwear'},
    {'color': [0, 53, 24], 'isthing': 1, 'id': 455, 'lvis_id': 1160, 'name': 'vacuum_cleaner'},
    {'color': [78, 116, 174], 'isthing': 1, 'id': 456, 'lvis_id': 1167, 'name': 'violin'},
    {'color': [10, 183, 214], 'isthing': 1, 'id': 457, 'lvis_id': 1169, 'name': 'volleyball'},
    {'color': [224, 8, 107], 'isthing': 1, 'id': 458, 'lvis_id': 1172, 'name': 'waffle_iron'},
    {'color': [112, 129, 251], 'isthing': 1, 'id': 459, 'lvis_id': 1175, 'name': 'walking_stick'},
    {'color': [102, 193, 120], 'isthing': 1, 'id': 460, 'lvis_id': 1178, 'name': 'wallet'},
    {'color': [110, 183, 168], 'isthing': 1, 'id': 461, 'lvis_id': 1179, 'name': 'walrus'},
    {'color': [209, 40, 143], 'isthing': 1, 'id': 462, 'lvis_id': 1180, 'name': 'wardrobe'},
    {'color': [21, 145, 71], 'isthing': 1, 'id': 463, 'lvis_id': 1182, 'name': 'automatic_washer'},
    {'color': [50, 21, 83], 'isthing': 1, 'id': 464, 'lvis_id': 1183, 'name': 'watch'},
    {'color': [101, 3, 174], 'isthing': 1, 'id': 465, 'lvis_id': 1184, 'name': 'water_bottle'},
    {'color': [83, 125, 52], 'isthing': 1, 'id': 466, 'lvis_id': 1186, 'name': 'water_faucet'},
    {'color': [208, 84, 71], 'isthing': 1, 'id': 467, 'lvis_id': 1187, 'name': 'water_filter'},
    {'color': [206, 164, 124], 'isthing': 1, 'id': 468, 'lvis_id': 1189, 'name': 'water_jug'},
    {'color': [117, 194, 62], 'isthing': 1, 'id': 469, 'lvis_id': 1191, 'name': 'water_scooter'},
    {'color': [195, 66, 150], 'isthing': 1, 'id': 470, 'lvis_id': 1192, 'name': 'water_ski'},
    {'color': [186, 41, 65], 'isthing': 1, 'id': 471, 'lvis_id': 1195, 'name': 'watermelon'},
    {'color': [205, 204, 190], 'isthing': 1, 'id': 472, 'lvis_id': 1201, 'name': 'wheel'},
    {'color': [237, 27, 1], 'isthing': 1, 'id': 473, 'lvis_id': 1202, 'name': 'wheelchair'},
    {'color': [2, 61, 143], 'isthing': 1, 'id': 474, 'lvis_id': 1207, 'name': 'wig'},
    {'color': [10, 220, 17], 'isthing': 1, 'id': 475, 'lvis_id': 1208, 'name': 'wind_chime'},
    {'color': [241, 106, 146], 'isthing': 1, 'id': 476, 'lvis_id': 1211, 'name': 'windshield_wiper'},
    {'color': [198, 19, 244], 'isthing': 1, 'id': 477, 'lvis_id': 1213, 'name': 'wine_bottle'},
    {'color': [104, 108, 38], 'isthing': 1, 'id': 478, 'lvis_id': 1215, 'name': 'wineglass'},
    {'color': [154, 154, 230], 'isthing': 1, 'id': 479, 'lvis_id': 1220, 'name': 'wooden_spoon'},
    {'color': [45, 160, 207], 'isthing': 1, 'id': 480, 'lvis_id': 1222, 'name': 'wrench'},
    {'color': [110, 109, 175], 'isthing': 1, 'id': 481, 'lvis_id': 1225, 'name': 'yacht'},
    {'color': [50, 109, 206], 'isthing': 1, 'id': 482, 'lvis_id': 1229, 'name': 'zebra'},
]


COMMON_BURST_CATEGORIES = [
    
]

UNCOMMON_BURST_CATEGORIES = [
    
]


def _get_burst_instances_meta():
    # NOTE: meta info
    lvis_ids = [k["lvis_id"] for k in ALL_BURST_CATEGORIES]
    colors = [k["color"] for k in ALL_BURST_CATEGORIES]
    # Mapping from the incontiguous LVIS category id to an id in [0, 481]
    lvis_id_to_contiguous_id = {k: i for i, k in enumerate(lvis_ids)}
    classes = [k["name"] for k in ALL_BURST_CATEGORIES]
    name_to_contigous_id = {k: i for i, k in enumerate(classes)}
    ret = {
        "name_to_contiguous_id": name_to_contigous_id,
        "thing_dataset_id_to_contiguous_id": lvis_id_to_contiguous_id,
        "thing_classes": classes,
        "thing_colors": colors,
    }
    return ret


def load_burst_json(json_file, image_root, dataset_name):
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with open(json_file, 'r') as fh:
        content = json.load(fh)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    meta = MetadataCatalog.get(dataset_name)

    # NOTE: format data
    sequences = content['sequences']

    dataset_dicts = []

    for sequence in sequences:
        annotations = []
        segms = sequence['segmentations']
        track_cat_ids = sequence['track_category_ids']
        for segm in segms:
            single_frame_annotations = []
            for track_id, anno in segm.items():
                single_frame_annotations.append({
                    'iscrowd': 0,
                    'category_id': meta.thing_dataset_id_to_contiguous_id[track_cat_ids[track_id]],
                    'id': int(track_id),
                    'bbox': [0., 0., 0., 0.],
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'segmentation': {'size': [sequence['height'], sequence['width']], 'counts': anno['rle']},   
                })
            annotations.append(single_frame_annotations)

        dataset_dicts.append({
            'width': sequence['width'],
            'height': sequence['height'],
            'length': len(sequence['annotated_image_paths']),
            # 'id': sequence['id'],
            'dataset': sequence['dataset'],
            'seq_name': sequence['seq_name'],
            'annotated_image_paths': sequence['annotated_image_paths'],
            'file_names': [os.path.join(image_root, sequence['dataset'], sequence['seq_name'], x) for x in sequence['annotated_image_paths']],
            'annotations': annotations,
            # 'segmentations': sequence['segmentations'],
            # 'track_category_ids': sequence['track_category_ids'],
        })
    return dataset_dicts


def register_burst_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in BURST's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_burst_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="burst", **metadata
    )


# ==== Predefined splits for BURST ===========
_PREDEFINED_SPLITS_BURST = {
    "burst_train": ("burst/frames/train",
                    "burst/annotations/train/train.json"),
    "burst_val":   ("burst/frames/val",
                    "burst/annotations/val/all_classes.json"),
    "burst_test":  ("burst/frames/test",
                    "burst/annotations/test/all_classes.json"),
}


def register_all_burst(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BURST.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_burst_instances(
            key,
            _get_burst_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# Assume pre-defined datasets live in `./datasets`.
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_burst(_root)


if __name__ == "__main__":
    """
    Test the BURST json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels
    from PIL import Image

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get("burst_train")

    json_file = "./datasets/burst/annotations/train/train.json"
    image_root = "./datasets/burst/frames/train/"
    dicts = load_burst_json(json_file, image_root, "burst_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "burst-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    class CVisualizer(Visualizer):
        
        def draw_dataset_dict(self, dic):
            """
            Draw annotations/segmentaions in Detectron2 Dataset format.

            Args:
                dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

            Returns:
                output (VisImage): image object with visualizations.
            """
            annos = dic.get("annotations", None)
            if annos:
                if "segmentation" in annos[0]:
                    masks = [x["segmentation"] for x in annos]
                else:
                    masks = None

                colors = None
                category_ids = [x["category_id"] for x in annos]
                if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                    colors = [
                        self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
                        for c in category_ids
                    ]
                names = self.metadata.get("thing_classes", None)
                labels = _create_text_labels(
                    category_ids,
                    scores=None,
                    class_names=names,
                    is_crowd=[x.get("iscrowd", 0) for x in annos],
                )
                self.overlay_instances(
                    labels=labels, boxes=None, masks=masks, keypoints=None, assigned_colors=colors
                )
            return self.output

    # for d in dicts:
    #     vid_name = d['dataset'] + '_' + d['seq_name']
    #     os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
    #     for idx, file_name in enumerate(d["annotated_image_paths"]):
    #         img_path = os.path.join(image_root, d['dataset'], d['seq_name'], file_name)
    #         img = np.array(Image.open(img_path))
    #         # BUG: Set boxes as None in `draw_dataset_dict` function in `Visualizer` class
    #         visualizer = CVisualizer(img, metadata=meta, instance_mode=ColorMode.SEGMENTATION)
    #         frame_dic = extract_frame_dic(d, idx)
    #         vis = visualizer.draw_dataset_dict(frame_dic)
    #         fpath = os.path.join(dirname, vid_name, file_name)
    #         vis.save(fpath)

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            # BUG: Set boxes as None in `draw_dataset_dict` function in `Visualizer` class
            visualizer = CVisualizer(img, metadata=meta, instance_mode=ColorMode.SEGMENTATION)
            frame_dic = extract_frame_dic(d, idx)
            vis = visualizer.draw_dataset_dict(frame_dic)
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
