import argparse
from tool.calculate_cofficient import standar_gaussian

BATCH_SIZE = 1
NUM_WORKERS = 12
GPU = 0
SEED = 20
BACKGROUND_THRESOLD = 0.1

NUM_STEPS = 50000
SAVE_PRED_EVERY = 1000

EXP_ROOT_DIR = 'WDA_cvk_BIBM/'

INPUT_SIZE = (512, 512)

DATA_DIRECTORY_IMG = './data/50%vncdata/train/img/'
DATA_DIRECTORY_LABEL = './data/50%vncdata/train/lab/'
DATA_LIST_PATH = './dataset/vncdata_list/train.txt'

INPUT_SIZE_TARGET = '512,512'
DATA_DIRECTORY_TARGET = './data/cvlabdata/train/img/'
DATA_DIRECTORY_TARGET_LABEL = './data/cvlabdata/train/lab/'
DATA_DIRECTORY_TARGET_PSEUDO_PARTIA_LABEL = ''
DATA_DIRECTORY_TARGET_POINT = './data/cvlabdata/train/15%_split1'
DATA_LIST_PATH_TARGET = './dataset/cvlabdata_list/train.txt'

# target validation
DATA_DIRECTORY_VAL = './data/cvlabdata/test/img/'
DATA_DIRECTORY_VAL_LABEL = './data/cvlabdata/test/lab/'
DATA_DIRECTORY_TARGET_VAL_DET = './data/cvlabdata/test/lab//'
DATA_LIST_PATH_VAL = './dataset/cvlabdata_list/test.txt'

DATA_DIRECTORY_TEST = './data/cvlabdata/test/img/'
DATA_DIRECTORY_TEST_LABEL = './data/cvlabdata/test/lab/'
DATA_LIST_PATH_TEST =  './dataset/cvlabdata_list/test.txt'



ITER_START = 1
PRETRAIN = 1

RESTORE_FROM = './pretrain_model/vnc_full_supervised.pth'
CountingModel_Path = './pretrain_model/vnc_count.pth'
D_RESTORE_FROM = ''
NUM_CLASSES = 2

LEARNING_RATE = 0.00005
STEP_SIZE = 5000
LEARNING_RATE_Dl = 0.0002
STEP_SIZE_Dl = 2000

SAVE_NUM_IMAGES = 2

SNAPSHOT_DIR = './step1_snapshots/'
ADV_WEIGHT = 0.5
DETECTION_WEIGHT = 0.1
BEST_TJAC = 0.10
BEST_MAE = 10

COFFICIENT = standar_gaussian(61)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir-img", type=str, default=DATA_DIRECTORY_IMG,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-label", type=str, default=DATA_DIRECTORY_LABEL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-test", type=str, default=DATA_DIRECTORY_TEST,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-test-label", type=str, default=DATA_DIRECTORY_TEST_LABEL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-val", type=str, default=DATA_DIRECTORY_VAL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-val-label", type=str, default=DATA_DIRECTORY_VAL_LABEL,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-dir-target-label", type=str, default=DATA_DIRECTORY_TARGET_LABEL,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-dir-target-pseudo-partial-label", type=str, default=DATA_DIRECTORY_TARGET_PSEUDO_PARTIA_LABEL,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-val", type=str, default=DATA_LIST_PATH_VAL,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-test", type=str, default=DATA_LIST_PATH_TEST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--step-size", type=int, default=STEP_SIZE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--det-weight", type=float, default=DETECTION_WEIGHT,
                        help="detection_weight for reconstruction training.")
    parser.add_argument("--adv-weight", type=float, default=ADV_WEIGHT,
                        help="adv_weight for reconstruction training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--iter-start", type=int, default=ITER_START,
                        help="Number of training steps.")
    parser.add_argument("--pretrain", type=int, default=PRETRAIN,
                        help="whether to pretrain the model.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--Drestore-from", type=str, default=D_RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--best_tjac", type=float, default=BEST_TJAC,
                        help="The best tjac")
    parser.add_argument("--best_mae", type=float, default=BEST_MAE,
                        help="The best MAE")
    parser.add_argument("--data-dir-target-point", type=str, default=DATA_DIRECTORY_TARGET_POINT,
                        help="Path to the directory containing the target dataset.")

    parser.add_argument("--step1-best-seg-val", type=str, default=DATA_DIRECTORY_TARGET_POINT,
                        help=".")
    parser.add_argument("--step1-best-det-val", type=str, default=DATA_DIRECTORY_TARGET_POINT,
                        help=".")
    parser.add_argument("--step2-best-seg-val", type=str, default=DATA_DIRECTORY_TARGET_POINT,
                        help=".")
    parser.add_argument("--step2-best-det-val", type=str, default=DATA_DIRECTORY_TARGET_POINT,
                        help=".")
    parser.add_argument("--gtpoint-partiallab", type=str, default='./gtpoint_partiallabel',
                        help=".")
    parser.add_argument("--seed", type=str, default=SEED, help=".")
    parser.add_argument("--learning-rate-Dl", type=float, default=LEARNING_RATE_Dl,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--step-size-Dl", type=int, default=STEP_SIZE_Dl,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--bg-thresold", type=int, default=BACKGROUND_THRESOLD,
                        help=".")
    parser.add_argument("--countingmodel_path", type=str, default=CountingModel_Path,
                        help="Where restore model parameters from.")

    # parser.add_argument("--oneshot-data-dir-target", type=str, default=ONESHOT_DATA_DIRECTORY_TARGET,
    #                     help="Path to the directory containing the target dataset.")
    # parser.add_argument("--oneshot-data-dir-target_label", type=str, default=ONESHOT_DATA_DIRECTORY_TARGET_LABEL,
    #                     help="Path to the directory containing the target dataset.")
    # parser.add_argument("--oneshot-data-dir-target_point", type=str, default=ONESHOT_DATA_DIRECTORY_TARGET_POINT,
    #                     help="Path to the directory containing the target dataset.")
    # parser.add_argument("--oneshot-data-list", type=str, default=ONESHOT_DATA_LIST_PATH,
    #                     help="Path to the file listing the images in the source dataset.")
    # parser.add_argument("--oneshot-data-dir-target_pseudo", type=str, default=ONESHOT_DATA_DIRECTORY_TARGET_PSEUDO,
    #                     help="Path to the directory containing the target dataset.")

    parser.add_argument("--data-dir-target_val_det", type=str, default=DATA_DIRECTORY_TARGET_VAL_DET,
                        help="Path to the directory containing the target dataset.")

    parser.add_argument("--cofficient", type=str, default=COFFICIENT,
                        help="the count result will divided by this cofficient.")

    parser.add_argument("--exp-root-dir", type=str, default=EXP_ROOT_DIR,
                        help="Where the experiment root dir .")


    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
