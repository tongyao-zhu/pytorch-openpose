import cv2
import copy
import numpy as np
import os
import json
import argparse

from src import model
from src import util
from src.body import Body
from src.hand import Hand
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    print("Error when setting start method")
    pass
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')


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


def get_batches(image_name_list, batch_size):
    batches = []
    batch_names = []
    for i in range(0, len(image_name_list), batch_size):
        batches.append(image_name_list[i:i + batch_size])
        batch_names.append(i // batch_size)
    return zip(batch_names, batches)


def get_batch_points(arguments):
    batch_name, image_name_list = arguments
    candidate_dict = {}
    subset_dict = {}
    hands_dict = {}
    all_hand_peaks_dict = {}
    print("start to process images")
    for input_image in image_name_list:
        # print(f"processing current image {count}")
        oriImg = cv2.imread(input_image)  # B,G,R order
        candidate, subset = body_estimation(oriImg)
        # print("candidate is {}".format(candidate))
        # print("subset is {}".format(subset))
        #
        # detect hand
        hands_list = util.handDetect(candidate, subset, oriImg)
        # print("hands list is {}".format(hands_list))
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
            peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            # else:
            #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
            #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
            #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            #     print(peaks)
            all_hand_peaks.append(peaks.tolist())

        candidate_dict[input_image] = candidate.tolist()
        subset_dict[input_image] = subset.tolist()
        hands_dict[input_image] = hands_list
        all_hand_peaks_dict[input_image] = all_hand_peaks
        #
        # print("all hands peak is {}".format(all_hand_peaks))
        # print("finished processing current image {}".format(input_image))
    filename = f"./keypoint_data/batch_{batch_name:03}.json"
    total_dict = {"candidates": candidate_dict, "subset": subset_dict, "hands": hands_dict,
                  "hand_peaks": all_hand_peaks_dict}
    with open(filename, 'w') as outfile:
        json.dump(total_dict, outfile)
    print(f"finished writing to json, the file name is {filename}")


def generate_points(image_name_list, save_batch_size, num_process):
    batches = list(get_batches(image_name_list, save_batch_size))
    print("Finished loading getting all batches, in total {} batches".format(len(batches)))
    pool = Pool(num_process)
    pool.map(get_batch_points, batches)
    pool.close()
    pool.join()
    print("Finished process!")


# def generate_points(image_name_list, save_batch_size=1000):
#     count = 0
#     candidate_dict = {}
#     subset_dict = {}
#     hands_dict = {}
#     all_hand_peaks_dict = {}
#     print("start to process images")
#     for input_image in image_name_list:
#         count += 1
#         # print(f"processing current image {count}")
#         oriImg = cv2.imread(input_image)  # B,G,R order
#         candidate, subset = body_estimation(oriImg)
#         # print("candidate is {}".format(candidate))
#         # print("subset is {}".format(subset))
#         #
#         # detect hand
#         hands_list = util.handDetect(candidate, subset, oriImg)
#         # print("hands list is {}".format(hands_list))
#         all_hand_peaks = []
#         for x, y, w, is_left in hands_list:
#             # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
#             # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#             # if is_left:
#             # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
#             # plt.show()
#             peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
#             peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
#             peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
#             # else:
#             #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
#             #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
#             #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
#             #     print(peaks)
#             all_hand_peaks.append(peaks.tolist())
#
#         candidate_dict[input_image] = candidate.tolist()
#         subset_dict[input_image] = subset.tolist()
#         hands_dict[input_image] = hands_list
#         all_hand_peaks_dict[input_image] = all_hand_peaks
#         #
#         # print("all hands peak is {}".format(all_hand_peaks))
#         # print("finished processing current image {}".format(input_image))
#         if (count) % save_batch_size == 0 or count == len(image_name_list) - 1:
#             print("finished processing {} images".format("count"))
#             filename = f"./keypoint_data/batch_{(count//save_batch_size):03}.json"
#             total_dict = {"candidates":candidate_dict, "subset":subset_dict, "hands":hands_dict, "hand_peaks":all_hand_peaks_dict}
#             with open(filename, 'w') as outfile:
#                 json.dump(total_dict, outfile)
#             print(f"finished writing to json, the file name is {filename}")
#             candidate_dict = {}
#             subset_dict = {}
#             hands_dict = {}
#             all_hand_peaks_dict = {}
#             assert len(candidate_dict)==0 and len(subset_dict)==0 and len(hands_dict)==0 and len(all_hand_peaks_dict)==0
#     print("Finished generating points of all images")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--image_list", type=str, default=None, help="file that contain all images")
    ap.add_argument("--corpus_path", type=str, default=None, help="the path to corpus containing video names")
    ap.add_argument("--batch_size", type=int, default=1000, help="the batch size to save the results")
    ap.add_argument("--num_process", type=int, default=15, help="number of processes we are using")

    args = ap.parse_args()
    print(f"Input arguments are {args}")
    if args.image_list:
        print("image list specified, therefore directly generating points")
        generate_points(args.image_list, args.batch_size, args.num_process)
    elif args.corpus_path:
        print("only corpus path is specified, starting from corpus path")
        image_list = read_image_list(args.corpus_path)
        print("get all images, in total {} images".format(len(image_list)))
        generate_points(image_list, args.batch_size, args.num_process)
    else:
        print("must specified one of image list or corpus path")
