import math
import os
import random
from copy import copy
import cv2
import datetime
import time
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Process

TRANSFORMATIONS = 120  # Usare preferibilmente multipli del PROCESSES
PROCESSES = 12


def data_augmentation(foreground, backgrounds, transformations) -> int:
    local_date = datetime.datetime.now()
    new_dir_path = 'output/' + str(local_date)
    os.mkdir(new_dir_path)
    for i in range(0, transformations, 1):
        index = random.randint(0, len(backgrounds) - 1)
        background = copy(backgrounds[index])

        if background.shape[0] < foreground.shape[0] or background.shape[1] < foreground.shape[1]:
            print("ERROR: the foreground exceeds the background dimensions")
            return 1

        if background.shape[0] - foreground.shape[0] == 0:
            row = 0
        else:
            row = random.randint(0, background.shape[0] - foreground.shape[0] - 1)

        if background.shape[1] - foreground.shape[1] == 0:
            col = 0
        else:
            col = random.randint(0, background.shape[1] - foreground.shape[1] - 1)

        alpha = random.randint(128, 255)
        alpha_correction_factor = float(alpha) / 255.0
        beta_correction_factor = 1.0 - alpha_correction_factor

        for j in range(0, foreground.shape[0], 1):
            for k in range(0, foreground.shape[1], 1):
                f_pixel = foreground[j, k]
                b_pixel = background[row + j, col + k]
                f_alpha = float(f_pixel[3]) / 255.0
                if f_alpha > 0.9:
                    background[row + j, col + k] = [float(b_pixel[0]) * beta_correction_factor + float(
                        f_pixel[0]) * alpha_correction_factor * f_alpha,
                                                    float(b_pixel[1]) * beta_correction_factor + float(
                                                        f_pixel[1]) * alpha_correction_factor * f_alpha,
                                                    float(b_pixel[2]) * beta_correction_factor + float(
                                                        f_pixel[2]) * alpha_correction_factor * f_alpha, 255]

        cv2.imwrite("output/" + str(local_date) + "/quokka_" + str(i) + ".png", background)

    return 0


def data_augmentation_multiprocessing(foreground, backgrounds, dir_path, trasformations):
    for i in range(0, trasformations, 1):
        index = random.randint(0, len(backgrounds) - 1)
        background = copy(backgrounds[index])

        if background.shape[0] < foreground.shape[0] or background.shape[1] < foreground.shape[1]:
            print("ERROR: the foreground exceeds the background dimensions")
            return 1

        if background.shape[0] - foreground.shape[0] == 0:
            row = 0
        else:
            row = random.randint(0, background.shape[0] - foreground.shape[0] - 1)

        if background.shape[1] - foreground.shape[1] == 0:
            col = 0
        else:
            col = random.randint(0, background.shape[1] - foreground.shape[1] - 1)

        alpha = random.randint(128, 255)
        alpha_correction_factor = float(alpha) / 255.0
        beta_correction_factor = 1.0 - alpha_correction_factor

        for j in range(0, foreground.shape[0], 1):
            for k in range(0, foreground.shape[1], 1):
                f_pixel = foreground[j, k]
                b_pixel = background[row + j, col + k]
                f_alpha = float(f_pixel[3]) / 255.0
                if f_alpha > 0.9:
                    background[row + j, col + k] = [float(b_pixel[0]) * beta_correction_factor + float(
                        f_pixel[0]) * alpha_correction_factor * f_alpha,
                                                    float(b_pixel[1]) * beta_correction_factor + float(
                                                        f_pixel[1]) * alpha_correction_factor * f_alpha,
                                                    float(b_pixel[2]) * beta_correction_factor + float(
                                                        f_pixel[2]) * alpha_correction_factor * f_alpha, 255]

        cv2.imwrite(dir_path + "/quokka_" + multiprocessing.current_process().name + "_" + str(i) + ".png", background)


def data_augmentation_multiprocessing_one_at_a_time(foreground, background_img, dir_path, count):
    background = copy(background_img)
    if background.shape[0] < foreground.shape[0] or background.shape[1] < foreground.shape[1]:
        print("ERROR: the foreground exceeds the background dimensions")
        return 1

    if background.shape[0] - foreground.shape[0] == 0:
        row = 0
    else:
        row = random.randint(0, background.shape[0] - foreground.shape[0] - 1)

    if background.shape[1] - foreground.shape[1] == 0:
        col = 0
    else:
        col = random.randint(0, background.shape[1] - foreground.shape[1] - 1)

    alpha = random.randint(128, 255)
    alpha_correction_factor = float(alpha) / 255.0
    beta_correction_factor = 1.0 - alpha_correction_factor

    for j in range(0, foreground.shape[0], 1):
        for k in range(0, foreground.shape[1], 1):
            f_pixel = foreground[j, k]
            b_pixel = background[row + j, col + k]
            f_alpha = float(f_pixel[3]) / 255.0
            if f_alpha > 0.9:
                background[row + j, col + k] = [float(b_pixel[0]) * beta_correction_factor + float(
                    f_pixel[0]) * alpha_correction_factor * f_alpha,
                                                float(b_pixel[1]) * beta_correction_factor + float(
                                                    f_pixel[1]) * alpha_correction_factor * f_alpha,
                                                float(b_pixel[2]) * beta_correction_factor + float(
                                                    f_pixel[2]) * alpha_correction_factor * f_alpha, 255]

    cv2.imwrite(dir_path + "/quokka_" + str(count) + ".png", background)


if __name__ == '__main__':

    # LOAD THE IMAGES
    background_path = "input/background/"
    foreground_path = "input/foreground/"

    backgrounds = []
    for b_path in os.listdir(background_path):
        image = cv2.imread(background_path + b_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            backgrounds.append(image)

    if len(backgrounds) == 0:
        print("ERROR: no files found at: " + background_path)

    f_path = os.listdir(foreground_path)[0]
    foreground = cv2.imread(foreground_path + f_path, cv2.IMREAD_UNCHANGED)
    if foreground is not None:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
    else:
        print("ERROR: foreground not found at: " + foreground_path)

    # SEQUENTIAL IMAGE COMPOSITION
    print("START Sequential Algorithm")

    start = time.time()
    result = data_augmentation(foreground, backgrounds, TRANSFORMATIONS)
    end = time.time()

    if result == 0:
        print("Program Sequential Successfully Completed")
    else:
        print("Program Sequential Failed")

    print(f'Sequential running took {end - start} seconds.')

    # JOBLIB MULTIPROCESSING IMAGE COMPOSITION
    print("START Multiprocessing Joblib Algorithm")

    start = time.time()
    local_date = datetime.datetime.now()
    new_dir_path = 'output/' + str(local_date)
    os.mkdir(new_dir_path)
    index = random.randint(0, len(backgrounds) - 1)
    background = backgrounds[index]
    Parallel(n_jobs=PROCESSES)(
        delayed(data_augmentation_multiprocessing_one_at_a_time)(foreground, background, new_dir_path, i) for i
        in range(TRANSFORMATIONS))
    end = time.time()

    print(f'Multiprocessing Joblib Running took {end - start} seconds.')

    # MULTIPROCESSING IMAGE COMPOSITION
    print("START Pool Multiprocessing Algorithm")

    # pool_size = multiprocessing.cpu_count() * 2
    pool_size = PROCESSES
    print("Using " + str(pool_size) + " processes")
    pool = multiprocessing.Pool(
        processes=pool_size,
    )
    trasformations_for_process = math.ceil(TRANSFORMATIONS / pool_size)
    start = time.time()
    local_date = datetime.datetime.now()
    new_dir_path = 'output/' + str(local_date)
    os.mkdir(new_dir_path)
    # prepare arguments iterable for pool.starmap
    args = [(foreground, backgrounds, new_dir_path, trasformations_for_process) for i in range(pool_size)]
    pool.starmap(data_augmentation_multiprocessing, args)

    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks

    end = time.time()
    print(f'Pool Multiprocessing Running took {end - start} seconds.')

    print("START Multiprocessing Algorithm")
    print("Using " + str(PROCESSES) + " processes")

    trasformations_for_process = math.ceil(TRANSFORMATIONS / PROCESSES)
    local_date = datetime.datetime.now()
    new_dir_path = 'output/' + str(local_date)
    os.mkdir(new_dir_path)
    processes = [Process(target=data_augmentation_multiprocessing,
                         args=(foreground, backgrounds, new_dir_path, trasformations_for_process,)) for i in
                 range(PROCESSES)]
    start = time.time()
    [p.start() for p in processes]
    [p.join() for p in processes]
    end = time.time()

    print(f'Multiprocessing Running took {end - start} seconds.')
