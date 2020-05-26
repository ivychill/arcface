
import os
from pathlib import Path
from time import time
from shutil import copyfile, copytree, rmtree, move
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

def gen_list_by_person_dir(base, src_dir, persons, dst_file):
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

def gen_list_by_person_file(base, src_dir, persons, dst_file):
    fd = open(base/dst_file, 'wt')
    label = 0
    persons = sorted(persons)
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
    gen_list_by_person_dir(base, src_dir, remain_persions, remain_file)

def minus_by_file(base, src_dir, minus_dir, src_file, minus_file, remain_file):
    src_persons = os.listdir(base/src_dir)
    minus_persons = os.listdir(base/minus_dir)
    remain_persons = set(src_persons) - set(minus_persons)
    gen_list_by_file(base, src_dir, src_file)
    gen_list_by_file(base, minus_dir, minus_file)
    gen_list_by_person_file(base, src_dir, remain_persons, remain_file)

def gen_dir_by_list(base, list_file, dst_dir):
    with open(base/list_file, 'r') as f:
        data_raw = f.readlines()
        for line_i in data_raw:
            line_i = line_i.strip('\n')
            path, pid = line_i.split(' ')
            dir, file = os.path.split(path)
            person = os.path.basename(dir)
            if not os.path.exists(base/dst_dir/person):
                print('copy person {}'.format(person))
                os.makedirs(base/dst_dir/person, exist_ok=True)
            copyfile(base/path, base/dst_dir/person/file)

def gen_dir_per_person_by_list(base, list_file, dst_dir):
    with open(base/list_file, 'r') as f:
        data_raw = f.readlines()
        pid = 100000
        for line_i in data_raw:
            line_i = line_i.strip('\n')
            path, _ = line_i.split(' ')
            dir, file = os.path.split(path)
            print('copy person {}'.format(str(pid)))
            os.makedirs(base/dst_dir/str(pid), exist_ok=True)
            copyfile(base/path, base/dst_dir/str(pid)/file)
            pid += 1

def mv(base, src_dir, dst_dir):
    persons = os.listdir(base/src_dir)
    persons.sort()
    for person in persons:
        # print('move person {}'.format(person))
        move(base/src_dir/person, base/dst_dir/person)

if __name__ == '__main__':
    base = Path('/srv/dataset')
    # base = Path('/home/kcadmin/user/fengchen/face/dataset/train')
    # src_dir = 'shanghai_cam_dynamic_112'
    # dst_file = 'shanghai_cam_dynamic_112.txt'
    # gen_list_by_dir(base, src_dir, dst_file)

    # src_dir = 'shanghai_cam_dynamic_112'
    # minus_dir = 'shanghai_cam_dynamic_112_test_1k'
    # src_file = 'shanghai_cam_dynamic_112.txt'
    # minus_file = 'shanghai_cam_dynamic_112_test_1k.txt'
    # remain_file = 'shanghai_cam_dynamic_112_train_6k.txt'
    # minus(base, src_di42w_112r, minus_dir, src_file, minus_file, remain_file)

    # src_dir = '42w_112/shanghai_important_42w'
    # minus_dir = '10w_112/shanghai_important_42w'
    # src_file = '42w_112.txt'
    # minus_file = '10w_112.txt'
    # remain_file = '32w_112.txt'
    # minus_by_file(base, src_dir, minus_dir, src_file, minus_file, remain_file)

    list_file = 'faces_emore.txt'
    dst_dir = 'faces_emore_16_per_peron/imgs'
    gen_dir_by_list(base, list_file, dst_dir)

    # list_file = '32w_112.txt'
    # dst_dir = '32w_112'
    # gen_dir_per_person_by_list(base, list_file, dst_dir)

    # src_dir = 'test/32w_112'
    # dst_dir = 'train/faces_emore_plus_sh_32w/imgs'
    # mv(base, src_dir, dst_dir)