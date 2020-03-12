import os
from shutil import copyfile, copytree, rmtree
import random
import numpy as np

def dir_per_person(base, src, dst):
    os.makedirs(os.path.join(base, dst), exist_ok=True)
    paths = os.listdir(os.path.join(base, src))
    for path in paths:
        print(path)
        if os.path.isdir(os.path.join(base, src, path)):
            print('copy dir')
            copytree(os.path.join(base, src, path), os.path.join(base, dst, path))
        else:
            print('copy file')
            name, ext = path.split('.')
            os.makedirs(os.path.join(base, dst, name), exist_ok=True)
            copyfile(os.path.join(base, src, path), os.path.join(base, dst, name, path))

def cp_intersection(base, src_from, src_to, dst_from, dst_to):
    src_persons = os.listdir(os.path.join(base, src_from))
    dst_persons = os.listdir(os.path.join(base, dst_from))
    inter_persions = set(src_persons) & set(dst_persons)
    for person in inter_persions:
        copytree(os.path.join(base, src_from, person), os.path.join(base, src_to, person))
        copytree(os.path.join(base, dst_from, person), os.path.join(base, dst_to, person))
    # rm_src_persons = set(src_persons) - inter_persions
    # for person in rm_src_persons:
    #     print('src:', person)
    #     os.remove(os.path.join(base, src, person))
    # rm_dst_persons = set(dst_persons) - inter_persions
    # for person in rm_dst_persons:
    #     print('dst:', person)
    #     os.remove(os.path.join(base, src, person))

def rm_except_intersection(base, src, dst):
    src_persons = os.listdir(os.path.join(base, src))
    dst_persons = os.listdir(os.path.join(base, dst))
    inter_persions = set(src_persons) & set(dst_persons)
    rm_src_persons = set(src_persons) - inter_persions
    for person in rm_src_persons:
        print('src:', person)
        rmtree(os.path.join(base, src, person))
    rm_dst_persons = set(dst_persons) - inter_persions
    for person in rm_dst_persons:
        print('dst:', person)
        rmtree(os.path.join(base, dst, person))

def select_nth_dir(base, src_from, src_to, dst_from, dst_to, n):
    if os.path.exists(os.path.join(base, src_to)):
        rmtree(os.path.join(base, src_to))
    if os.path.exists(os.path.join(base, dst_to)):
        rmtree(os.path.join(base, dst_to))
    src_persons = os.listdir(os.path.join(base, src_from))
    random.shuffle(src_persons)
    select_persons = src_persons[::n]

    for person in select_persons:
        print(person)
        copytree(os.path.join(base, src_from, person), os.path.join(base, src_to, person))
        copytree(os.path.join(base, dst_from, person), os.path.join(base, dst_to, person))

def select_nth_file(base, src, dst, n):
    if os.path.exists(os.path.join(base, dst)):
        rmtree(os.path.join(base, dst))
    os.makedirs(os.path.join(base, dst), exist_ok=True)
    src_persons = os.listdir(os.path.join(base, src))
    random.shuffle(src_persons)
    print(src_persons)
    select_persons = src_persons[::n]
    print(select_persons)
    for person in select_persons:
        print(person)
        copyfile(os.path.join(base, src, person), os.path.join(base, dst, person))

def make_q_g(base, src, dst, n):
    persons = os.listdir(os.path.join(base, src))
    for person in persons:
        os.makedirs(os.path.join(base, dst, person), exist_ok=True)
        imgs = os.listdir(os.path.join(base, src, person))
        for img in imgs[:n]:
            print(img)
            os.rename(os.path.join(base, src, person, img), os.path.join(base, dst, person, img))

def rm_empty(base, subdir):
    persons = os.listdir(os.path.join(base, subdir))
    for person in persons:
        if os.path.isfile(os.path.join(base, subdir, person)):
            print('rm person ', person)
            os.remove(os.path.join(base, subdir, person))
        else:
            imgs = os.listdir(os.path.join(base, subdir, person))
            if len(imgs) == 0:
                print('rm subdir {}/{}'.format(subdir, person))
                rmtree(os.path.join(base, subdir, person))

def rm_dirty_dir(base, subdir, dirty_list):
    dirty_persons = []
    with open(dirty_list, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        dirty_persons.append(line)

    rm_count = 0
    persons = os.listdir(os.path.join(base, subdir))
    for person in persons:
        if person in dirty_persons:
            print('rm subdir {}/{}'.format(subdir, person))
            rmtree(os.path.join(base, subdir, person))
            rm_count += 1
    print('rm {} dirty persons from {}'.format(rm_count, subdir))

def rm_dirty_file(base, subdir, dirty_list):
    dirty_persons = []
    with open(dirty_list, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        dirty_persons.append(line)

    rm_count = 0
    images = os.listdir(os.path.join(base, subdir))
    for image in images:
        name, ext = image.split('.')
        if name in dirty_persons:
            print('rm image {}/{}'.format(subdir, image))
            os.remove(os.path.join(base, subdir, image))
            rm_count += 1
    print('rm {} dirty persons from {}'.format(rm_count, subdir))

def split_train_test(base, src_from, src_to_train, src_to_test, dst_from, dst_to_train, dst_to_test, n):
    if os.path.exists(os.path.join(base, src_to_train)):
        rmtree(os.path.join(base, src_to_train))
    os.makedirs(os.path.join(base, src_to_train))
    if os.path.exists(os.path.join(base, src_to_test)):
        rmtree(os.path.join(base, src_to_test))
    os.makedirs(os.path.join(base, src_to_test))
    if os.path.exists(os.path.join(base, dst_to_train)):
        rmtree(os.path.join(base, dst_to_train))
    os.makedirs(os.path.join(base, dst_to_train))
    if os.path.exists(os.path.join(base, dst_to_test)):
        rmtree(os.path.join(base, dst_to_test))
    os.makedirs(os.path.join(base, dst_to_test))

    src_persons = os.listdir(os.path.join(base, src_from))
    random.shuffle(src_persons)

    test_size = int(np.floor(1.0/n * len(src_persons)))
    print('test_size:', test_size)

    test_persons = src_persons[:test_size]
    for person in test_persons:
        # print('test: ', person)
        copytree(os.path.join(base, src_from, person), os.path.join(base, src_to_test, person))
        copytree(os.path.join(base, dst_from, person), os.path.join(base, dst_to_test, person))

    train_persons = src_persons[test_size:]
    for person in train_persons:
        # print('train: ', person)
        copytree(os.path.join(base, src_from, person), os.path.join(base, src_to_train, person))
        copytree(os.path.join(base, dst_from, person), os.path.join(base, dst_to_train, person))

if __name__ == '__main__':
    # base = '/srv/dataset/test'
    base = '../dataset/test'
    # src = 'kc_employee_id'
    # dst = 'kc_id'
    # dir_per_person(base, src, dst)

    # src_from = 'kc_id'
    # src_to = 'kc_id_inter'
    # dst_from = 'kc_dynamic'
    # dst_to = 'kc_dynamic_inter'
    # cp_intersection(base, src_from, src_to, dst_from, dst_to)

    # src_from = 'shanghai_cam_id'
    # src_to = 'shanghai_cam_id_3rd'
    # dst_from = 'shanghai_cam_dynamic'
    # dst_to = 'shanghai_cam_dynamic_3rd'
    # select_nth_dir(base, src_from, src_to, dst_from, dst_to, 3)

    # src = 'g_lfw'
    # dst = 'q_lfw'
    # make_q_g(base, src, dst, 1)

    # src = 'q_lfw_112'
    # dst = 'g_lfw_112'
    # rm_empty(base, src)
    # rm_empty(base, dst)
    # rm_except_intersection(base, src, dst)

    # src = 'kc_employee_id_112'
    # dst = 'kc_employee_dynamic_112'
    # rm_empty(base, src)
    # rm_empty(base, dst)
    # rm_except_intersection(base, src, dst)

    # src = 'shanghai_cam_id_112'
    # dst = 'shanghai_cam_dynamic_112'
    # rm_empty(base, src)
    # rm_empty(base, dst)
    # rm_except_intersection(base, src, dst)

    # src = 'shanghai_cam_id_112'
    # dst = 'shanghai_cam_dynamic_112'
    # dirty_list = 'data/dirty_list.txt'
    # rm_dirty_dir(base, src, dirty_list)
    # rm_dirty_dir(base, dst, dirty_list)

    # src = '42w_112/shanghai_important_42w'
    # dirty_list = 'data/dirty_list.txt'
    # rm_dirty_file(base, src, dirty_list)

    # src = '42w_112/shanghai_important_42w'
    # dst = '10w_112/shanghai_important_42w'
    # select_nth_file(base, src, dst, 4)

    src_from = 'shanghai_cam_id_112'
    src_to_train = 'shanghai_cam_id_112_train_6k'
    src_to_test = 'shanghai_cam_id_112_test_1k'
    dst_from = 'shanghai_cam_dynamic_112'
    dst_to_train = 'shanghai_cam_dynamic_112_train_6k'
    dst_to_test = 'shanghai_cam_dynamic_112_test_1k'
    split_train_test(base, src_from, src_to_train, src_to_test, dst_from, dst_to_train, dst_to_test, 7)