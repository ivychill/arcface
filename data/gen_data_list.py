
import os
from pathlib import Path
from time import time
from shutil import copyfile, copytree, rmtree
import random
import numpy as np


def gen_list_by_dir(base, src_dir, dst_file):
    fd = open(base/dst_file, 'wt')
    persons = os.listdir(base/src_dir)
    persons.sort()
    label = 0
    for person in persons:
        print(person)
        if os.path.isdir(base/src_dir/person):
            images = os.listdir(base/src_dir/person)
            images.sort()
            for image in images:
                ext = os.path.splitext(image)[1]
                if ext in ['.jpg', '.png']:
                    fd.write('{} {}\n'.format(Path(src_dir)/person/image, label))
        label += 1

def gen_list_by_file(base, src_dir, dst_file):
    fd = open(base/dst_file, 'wt')
    persons = os.listdir(base/src_dir)
    persons.sort()
    label = 0
    for person in persons:
        print(person)
        ext = os.path.splitext(person)[1]
        if ext in ['.jpg', '.png']:
            fd.write('{} {}\n'.format(Path(src_dir)/person, label))
        label += 1

def gen_list_by_person_dir(base, persons, dst_file):
    fd = open(base/dst_file, 'wt')
    label = 0
    persons = sorted(persons)
    for person in persons:
        print(person)
        if os.path.isdir(base/src_dir/person):
            images = os.listdir(base/src_dir/person)
            images.sort()
            for image in images:
                ext = os.path.splitext(image)[1]
                if ext in ['.jpg', '.png']:
                    fd.write('{} {}\n'.format(Path(src_dir)/person/image, label))
        label += 1

def gen_list_by_person_file(base, persons, dst_file):
    fd = open(base/dst_file, 'wt')
    label = 0
    persons = sorted(person)
    for person in persons:
        print(persons)
        ext = os.path.splitext(person)[1]
        if ext in ['.jpg', '.png']:
            fd.write('{} {}\n'.format(Path(src_dir)/person, label))
        label += 1

def minus_by_dir(base, src_dir, minus_dir, src_file, minus_file, remain_file):
    src_persons = os.listdir(base/src_dir)
    minus_persons = os.listdir(base/minus_dir)
    remain_persions = set(src_persons) - set(minus_persons)
    gen_list_by_dir(base, src_dir, src_file)
    gen_list_by_dir(base, minus_dir, minus_file)
    gen_list_by_person_dir(base, remain_persions, remain_file)

def minus_by_file(base, src_dir, minus_dir, src_file, minus_file, remain_file):
    src_persons = os.listdir(base/src_dir)
    minus_persons = os.listdir(base/minus_dir)
    remain_persions = set(src_persons) - set(minus_persons)
    gen_list_by_file(base, src_dir, src_file)
    gen_list_by_file(base, minus_dir, minus_file)
    gen_list_by_person_file(base, remain_persions, remain_file)

if __name__ == '__main__':
    base = Path('/srv/dataset/test')
    # base = Path('/home/kcadmin/user/fengchen/face/dataset/test')
    # src_dir = 'shanghai_cam_dynamic_112'
    # dst_file = 'shanghai_cam_dynamic_112.txt'
    # gen_list_by_dir(base, src_dir, dst_file)

    # src_dir = 'shanghai_cam_dynamic_112'
    # minus_dir = 'shanghai_cam_dynamic_112_test_1k'
    # src_file = 'shanghai_cam_dynamic_112.txt'
    # minus_file = 'shanghai_cam_dynamic_112_test_1k.txt'
    # remain_file = 'shanghai_cam_dynamic_112_train_6k.txt'
    # minus(base, src_di42w_112r, minus_dir, src_file, minus_file, remain_file)

    src_dir = '42w_112/shanghai_important_42w'
    minus_dir = '10w_112/shanghai_important_42w'
    src_file = '42w_112.txt'
    minus_file = '10w_112.txt'
    remain_file = '32w_112.txt'
    minus_by_file(base, src_dir, minus_dir, src_file, minus_file, remain_file)