from torch.utils import data

from add_arguments import get_arguments
from dataset.target_dataset import targetDataSet_test
from utils.stats_utils import *
from model.HSC82 import PGS_Net, WDA_Net_r
from utils.stats_utils import *
from utils.tools_self import *
import torch.nn.functional as F
import xlsxwriter
from skimage import morphology

def evl_model(model, valloader, seg_save_dir, det_save_dir, gpu, usecuda):
    if usecuda:
        model.cuda(gpu)
    model.eval()

    total_dice = 0
    total_jac = 0
    total_aji = 0
    total_aji_plus = 0
    total_pq = 0
    jac_list=[]
    dice_list = []
    aji_list = []
    pq_list = []
    count = 0
    # test_augmentation = True
    test_augmentation = False
    for i_pic, (images_v, _, original_msk, points_v, _, name) in enumerate(valloader):

        with torch.no_grad():
            if usecuda:
                image_v = images_v.unsqueeze(0).cuda(gpu)
            else:
                image_v = images_v.unsqueeze(0)
            try:
                if test_augmentation == False:
                    seg_output, det_output, _ = model(image_v)
                    seg_output = torch.softmax(seg_output, dim=1)[:, 1, :, :].float().cpu()
                    seg_output = seg_output[:, (images_v.shape[1] - original_msk.shape[1]):,
                                 (images_v.shape[-1] - original_msk.shape[-1]):]
                    det_output = det_output[:,:, (images_v.shape[1] - original_msk.shape[1]):,
                                 (images_v.shape[-1] - original_msk.shape[-1]):]
                else:
                    seg_output, det_output, _ = model(image_v)
                    output = F.softmax(seg_output, dim=1)

                    seg_output, det_output, _ = model(image_v.flip(dims=(2,)))
                    seg_output = seg_output.flip(dims=(2,))
                    output += F.softmax(seg_output, dim=1)

                    seg_output, det_output, _ = model(image_v.flip(dims=(3,)))
                    seg_output = seg_output.flip(dims=(3,))
                    output += F.softmax(seg_output, dim=1)

                    seg_output, det_output, _ = model(image_v.flip(dims=(2, 3)))
                    seg_output = seg_output.flip(dims=(2, 3))
                    output += F.softmax(seg_output, dim=1)

                    output = output / 4.0
                    output = output[:, 1, :, :].float().cpu()
                    seg_output = output[:, (images_v.shape[1] - original_msk.shape[1]):,
                                 (images_v.shape[-1] - original_msk.shape[-1]):]
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e


        save_det_pred_image((det_output[0, 0, :, :]), name, det_save_dir)

        seg_cpu = seg_output.data[0].cpu().numpy()
        seg_cpu = polarize(seg_cpu)
        dice, jac = dice_coeff(seg_cpu, original_msk)
        seg_pred = save_seg_pred_image(seg_output, name, seg_save_dir, jac)

        aji = get_fast_aji(original_msk.cpu().numpy().astype("uint8"), seg_pred.astype("uint8"))
        aji_plus = get_fast_aji_plus(original_msk.cpu().numpy().astype("uint8"), seg_pred.astype("uint8"))
        pq, _, _, _, _ = get_fast_pq(original_msk.cpu().numpy().astype("uint8"), seg_pred.astype("uint8"))

        total_dice = total_dice + dice
        total_jac = total_jac + jac

        total_aji = total_aji + aji
        total_aji_plus = total_aji_plus + aji_plus

        total_pq = total_pq + pq[2]
        dice_list.append(('%3f'%dice))
        jac_list.append(('%3f'%jac))
        aji_list.append(('%3f'%aji))
        pq_list.append(('%3f'%pq[2]))
        count = count + 1
        print(count)
        print("dice:{0:.5f},jac = {1:.5f}".format(dice, jac))
        print("aji:{0:.5f},aji_plus = {1:.5f}".format(aji, aji_plus))
        print(pq)
    print('test dice: %4f' % (total_dice / count), 'test jac: %4f' % (total_jac / count), 'aji: %4f' % (total_aji / count), 'aji_plus: %4f' % (total_aji_plus / count), "pq: %4f" % (total_pq / count))

    return dice_list, jac_list, aji_list, pq_list


def save_seg_pred_image(input, im_name, save_folder_name, jac):

    img_cont_np = input.data.cpu().numpy()
    # img_cont_np = input.data[0].cpu().numpy()
    img_cont_np = polarize(img_cont_np)

    img_cont = (img_cont_np * 255).astype('uint8')

    save_name = os.path.join(save_folder_name, str(im_name[0]))


    cv2.imwrite(save_name, img_cont)

    return img_cont_np


def save_det_pred_image(input, im_name, save_folder_name):

    img_cont_np = input.data.cpu().numpy()
    img_cont_np = MaxMinNormalization(img_cont_np, np.max(img_cont_np), np.min(img_cont_np))

    img_cont = (img_cont_np* 255).astype('uint8')

    save_name = os.path.join(save_folder_name,  str(im_name[0]))
    cv2.imwrite(save_name, img_cont)

    return img_cont_np


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


def polarize(img):
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img



if __name__ == '__main__':

    setup_seed(20)

    args = get_arguments()
    root_save_dir = r'./test/'
    make_dirs(root_save_dir)
    sys.stdout = Logger(stream=sys.stdout, filename=root_save_dir+'testlog.log')
    sys.stderr = Logger(stream=sys.stderr, filename=root_save_dir+'testlog.log')

    args.data_dir_test = './data/cvlabdata/test/img/'
    args.data_dir_test_label = './data/cvlabdata/test/lab/'
    args.data_list_test = './dataset/cvlabdata_list/test.txt'


    makedatalist(args.data_dir_test, args.data_list_test)

    test_model_path = r''
    print(test_model_path)

    model = WDA_Net_r(in_channels=1, out_channels=2, device="cuda:" + str(args.gpu))
    model.load_state_dict(torch.load(test_model_path, map_location="cuda:" + str(args.gpu)), strict=False)

    input_size = (768, 1024)

    testloader = data.DataLoader(
        targetDataSet_test(args.data_dir_test, args.data_dir_test_label, args.data_list_test,
                           crop_size=input_size),
        batch_size=1, shuffle=False)


    test_exp_name = 'test'
    seg_output_dir = root_save_dir + test_exp_name + 'output/seg/'
    det_output_dir = root_save_dir + test_exp_name + 'output/det/'

    print(seg_output_dir)
    make_dirs(seg_output_dir)
    make_dirs(det_output_dir)

    dice, jac, aji, pq = evl_model(model, testloader, seg_output_dir,det_output_dir, args.gpu, usecuda=True)

































