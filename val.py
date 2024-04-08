from torch.utils import data

from add_arguments import get_arguments
from dataset.target_dataset import targetDataSet_test, kplusDataSet_test
from model.HSC82 import WDA_Net_r
from utils.stats_utils import *
from utils.tools_self import *
import config
from utils.tools_self import  *

from model.HSC82 import PGS_Net

cofficient = standar_gaussian(config.get_value())
##########
def val_count_model(model, valloader, gpu, usecuda):
    if usecuda:
        model.cuda(gpu)
    model.eval()
    count = 0
    total_error = 0
    gt_count = []
    for i_pic, (images_v, _, gt_num, name) in enumerate(valloader):
        with torch.no_grad():
            if usecuda:
                image_v = images_v.unsqueeze(0).cuda(gpu)
            else:
                image_v = images_v.unsqueeze(0)
            try:
                # _, output, _ = model(image_v)
                output= model(image_v)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
        su = torch.sum(output, dim=(2, 3)) / cofficient

        count_one = su.data.cpu().numpy()[0][0]
        count_one = np.round(count_one)
        gt = gt_num.cpu().numpy()
        error = np.abs(gt - count_one)
        total_error = total_error + error
        count = count+1
        gt_count.append(gt)
        # print(name, 'predict:', count_one, 'GT:', gt)

    #print('min:', min(gt_count), 'max:', max(gt_count), 'sum:', sum(gt_count))

    return total_error/count

##########
def validate_model(model, valloader, seg_save_dir, det_save_dir,  gpu, usecuda, type='cvlab'):

    model.cuda(gpu)
    model.eval()
    total_dice = 0
    total_jac = 0
    count = 0

    seg = torch.zeros((4096, 4096))
    det = torch.zeros((4096, 4096))
    mask = torch.zeros((4096, 4096))

    for i_pic, (images_v, _, original_msk, det_gt, name) in tqdm.tqdm(enumerate(valloader)):

        with torch.no_grad():
            if usecuda:
                image_v = images_v.unsqueeze(0).cuda(gpu)
            else:
                image_v = images_v.unsqueeze(0)
            try:
                seg_output, det_output, _, = model(image_v)
                seg_output = torch.softmax(seg_output, dim=1)[:, 1, :, :].float().cpu()

                seg_output = seg_output[:, (images_v.shape[1] - original_msk.shape[1]):,
                             (images_v.shape[-1] - original_msk.shape[-1]):]
                det_output = det_output[:, :, (images_v.shape[1] - original_msk.shape[1]):,
                             (images_v.shape[-1] - original_msk.shape[-1]):]
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

        if type != 'mito':
            seg_cpu = seg_output.data[0].cpu().numpy()
            seg_cpu = polarize(seg_cpu)
            dice, jac = dice_coeff(seg_cpu, original_msk)

            save_seg_pred_image(seg_output.data[0], name, seg_save_dir, jac)

            save_det_pred((det_output[0, 0, :, :]), name, det_save_dir)

            total_dice = total_dice + dice
            total_jac = total_jac + jac

            count = count + 1

        else:
            fromx = int(name[0].split("_")[1])
            endx = fromx + 1024
            fromy = int(name[0].split("_")[2].replace(".png", ""))
            endy = fromy + 1024

            name = (''.join(name))[:6]+'.png'
            newname = []
            newname.append(name)
            name = tuple(newname)
            seg[fromx:endx, fromy:endy] += seg_output.data[0]
            det[fromx:endx, fromy:endy] += det_output[0, 0, :, :].cpu()
            mask[fromx:endx, fromy:endy] += original_msk.data[0]
            if (i_pic+1) % 49 == 0:
                seg[512:3584, :] /= 2
                seg[:, 512:3584] /= 2

                det[512:3584, :] /= 2
                det[:, 512:3584] /= 2

                mask[512:3584, :] /= 2
                mask[:, 512:3584] /= 2
                mask = mask.cpu().numpy()
                mask[mask > 0] = 1

                seg_cpu = seg.cpu().numpy()
                seg_cpu = polarize(seg_cpu)
                dice, jac = dice_coeff(seg_cpu, mask)

                save_seg_pred_image(seg, name, seg_save_dir, jac)

                save_det_pred(det, name, det_save_dir)

                total_dice = total_dice + dice
                total_jac = total_jac + jac

                count = count + 1
                seg = torch.zeros((4096, 4096))
                det = torch.zeros((4096, 4096))
                mask = torch.zeros((4096, 4096))


    return total_dice / count, total_jac / count


def save_seg_pred_image(input, im_name, save_folder_name, jac):

    img_cont_np = input.data.cpu().numpy()
    img_cont_np = polarize(img_cont_np)

    img_cont = (img_cont_np * 255).astype('uint8')

    save_name = os.path.join(save_folder_name, str(im_name[0]))
    text = str('%.3f' % jac)
    # cv2.putText(img_cont, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    cv2.imwrite(save_name, img_cont)

    return img_cont_np

def save_pred(input, im_name, save_folder_name, original_msk):

    img_cont = (input ).astype('uint8')

    save_name = os.path.join(save_folder_name, str(im_name[0]))
    if not os.path.exists(save_name):
        print('save_name')
        cv2.imwrite(save_name, img_cont)

    return  original_msk

def save_det_pred(input, im_name, save_folder_name):

    img_cont_np = input.data.cpu().numpy()

    # img_cont_np[img_cont_np < -10] = -10
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

##########
def generate_pseudolabel_entroy(model, valloader, save_dir, det_save_path, gpu, usecuda):
    if usecuda:
        model.cuda(gpu)
    model.eval()
    make_dirs(save_dir)
    make_dirs(det_save_path)
    for i_pic, (images_v, _, original_msk, _, name) in tqdm.tqdm(enumerate(valloader)):

        with torch.no_grad():
            if usecuda:
                image_v = images_v.unsqueeze(0).cuda(gpu)
            else:
                image_v = images_v.unsqueeze(0)
            try:
                seg_output, det_output, _ = model(image_v)
                output = torch.softmax(seg_output, dim=1)
                det_output_flip = det_output

                seg_output, det_output, _ = model(image_v.flip(dims=(2,)))
                seg_output = seg_output.flip(dims=(2,))
                det_output = det_output.flip(dims=(2,))
                output += torch.softmax(seg_output, dim=1)
                det_output_flip += det_output

                seg_output, det_output, _ = model(image_v.flip(dims=(3,)))
                seg_output = seg_output.flip(dims=(3,))
                det_output = det_output.flip(dims=(3,))
                output += torch.softmax(seg_output, dim=1)
                det_output_flip += det_output

                seg_output, det_output, _ = model(image_v.flip(dims=(2, 3)))
                seg_output = seg_output.flip(dims=(2, 3))
                det_output = det_output.flip(dims=(2, 3))
                output += torch.softmax(seg_output, dim=1)
                det_output_flip += det_output

                output = output / 4.0
                det_output_flip = det_output_flip / 4.0

                output = output[:,:, (images_v.shape[1] - original_msk.shape[1]):,
                         (images_v.shape[-1] - original_msk.shape[-1]):]
                det_output = det_output_flip[:, :, (images_v.shape[1] - original_msk.shape[1]):,
                             (images_v.shape[-1] - original_msk.shape[-1]):]

                output = output.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
                #
                predicted_label = label.copy()
                predicted_entropy = compute_entropy(output, num_classes=2)

                thres = []
                for i in range(2):
                    x = predicted_entropy[predicted_label == i]
                    if len(x) == 0:
                        thres.append(0)
                        continue
                    x = np.sort(x)
                    # percentile_80 = np.percentile(x_sorted, 80)
                    thres.append(np.percentile(x, 80))
                thres = np.array(thres)
                thres[thres<0.5]=0.5
                for i in range(2):
                    label[(predicted_entropy > thres[i]) * (label == i)] = 255

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
            label = (label).astype('uint8')
            # label[label==255]=0
            save_name = os.path.join(save_dir, str(name[0]))
            cv2.imwrite(save_name, label)

            save_det_pred(det_output.data[0, 0, :, :], name, det_save_path)

##########
def val_main_entroy(val_path, save_path, det_save_path,args):
    setup_seed(20)

    # args = get_arguments()

    makedatalist(args.data_dir_target, args.data_list_target)

    test_model_path = val_path

    model = WDA_Net_r(in_channels=1, out_channels=2, device="cuda:" + str(args.gpu))
    model.load_state_dict(torch.load(test_model_path, map_location="cuda:" + str(args.gpu)),strict=False)

    input_size = (768, 1024)

    testloader = data.DataLoader(
        targetDataSet_test(args.data_dir_target, args.data_dir_target_label, args.data_list_target,
                          crop_size=input_size),
        batch_size=1, shuffle=False)

    generate_pseudolabel_entroy(model, testloader, save_path, det_save_path, args.gpu, usecuda=True)


if __name__ == '__main__':
   pass
