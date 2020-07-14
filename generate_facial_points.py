import cv2
import copy
import numpy as np
import os
import json
import argparse
import sys
import os
import dlib
import glob


def parts_to_points(parts):
    result = []
    for part in parts:
        result.append((part.x, part.y))
    return result


def get_image_paths(video_path):
    image_names = sorted(filter(lambda x: x[-4:] == ".png", os.listdir(video_path)))

    full_image_paths = list(map(lambda x: (video_path + x).replace("\n", ""), image_names))
    images_list = full_image_paths
    return images_list


def read_image_list(corpus_path):
    # processing corpus
    image_name_list = []
    for mode in ["dev", "test", "train"]:
        print("processing {} corpus".format(mode))
        old_dev_corpus = open(os.path.join(corpus_path, "phoenix2014T.{}.sign".format(mode)), "r").readlines()

        count = 0
        for folder_path in old_dev_corpus:
            count += 1
            if not count % (len(old_dev_corpus) // 10):
                print("currently processed {} of {} ({}%)".format(count, len(old_dev_corpus),
                                                                  count * 100 // len(old_dev_corpus)))
            real_path = (folder_path.replace("<PATH_TO_EXTRACTED_AND_RESIZED_FRAMES>",
                                             "./../PHOENIX-2014-T-release-v3/PHOENIX-2014-T")).replace("227x227",
                                                                                                       "210x260").replace(
                "\n", "")
            image_name_list.extend(get_image_paths(real_path))
        print('after the current section, count is {}'.format(count))

    with open('./image_name_list.txt', 'w') as f:
        for item in image_name_list:
            f.write("%s \n" % item)
    return image_name_list


def generate_facial_points(image_name_list, predictor_path, save_batch_size=1000):
    print("batch size is {}".format(save_batch_size))
    name_points_dict = {}
    count = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    for f in image_name_list:
        count += 1

        # win = dlib.image_window()

        # print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        # win.clear_overlay()
        # win.set_image(img)
        dets = detector(img, 1)
        # print("Number of faces detected: {} in".format(len(dets)))

        if len(dets)==0:
            name_points_dict[f]=[]

        try:
            assert len(dets) <= 1
        except:
            print("Error! More than 1 faces are identified in a single image")
            print("name of file is {}".format(f))
            name_points_dict[f]=[]
        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
            # d.bottom())) Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                     # shape.part(1)))

            # print("It has {} parts".format(shape.num_parts))
            assert shape.num_parts == 68
            # print("parts are {}".format(shape.parts()))

            name_points_dict[f] = parts_to_points(shape.parts())

            # print("Part 60 is {}".format(shape.part(60)))
            # Draw the face landmarks on the screen.
            # win.add_overlay(shape)
        # print("before saving, count is {}".format(count))
        if count % save_batch_size == 0 or count == len(image_name_list):
            print("finished processing {} images".format(count))
            if count == len(image_name_list):
                filename = f"./facial_data/batch_{(count // save_batch_size + 1):03}.json"
            else:
                filename = f"./facial_data/batch_{(count // save_batch_size):03}.json"
            with open(filename, 'w') as outfile:
                json.dump(name_points_dict, outfile)
            print(f"finished writing to json, the file name is {filename}")
            name_points_dict = {}
            assert len(name_points_dict) == 0

        # win.add_overlay(dets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--image_list", type=str, default=None, help="file that contain all images")
    ap.add_argument("--corpus_path", type=str, default=None, help="the path to corpus containing video names")
    ap.add_argument("--batch_size", type=int, default=1000, help="the batch size to save the results")
    ap.add_argument("--predictor_path", type=str, required=True, help="the path to the pretrained predictory model")
    args = ap.parse_args()
    print("Using CUDA: {}".format(dlib.DLIB_USE_CUDA))
    if args.image_list:
        print("image list specified, therefore directly generating points")
        generate_facial_points(args.image_list, args.predictor_path, args.batch_size)
    elif args.corpus_path:
        print("only corpus path is specified, starting from corpus path")
        image_list = read_image_list(args.corpus_path)
        print("get all images, in total {} images".format(len(image_list)))
        generate_facial_points(image_list, args.predictor_path, args.batch_size)
    else:
        print("must specified one of image list or corpus path")
