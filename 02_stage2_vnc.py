from imports import *
from model.HSC82 import WDA_Net_r
from model.discriminator import labelDiscriminator_seg
from val import val_main_entroy

args = get_arguments()
args.num_steps = 100000
args.countingmodel_path = './pretrain_model/vnc_count.pth'

def main(exp_dir, model_path, Dmodel_path,  step2_pseudoseg_save_path):

    print('============' + exp_dir + '============')
    args.snapshot_dir = os.path.join(exp_dir, 'snapshots')
    make_dirs(args.snapshot_dir)

    setup_seed(args.seed)

    sys.stdout = Logger(stream=sys.stdout, filename = exp_dir + '/trainlog.log')
    sys.stderr = Logger(stream=sys.stderr, filename = exp_dir + '/trainlog.log')

    print('============Generate GT Sparse Label Start============')
    gt_partiallabel_save_path = os.path.join(exp_dir, 'gt_partiallabel')
    generate_sparse_plabel(args.data_dir_target_point, args.data_dir_target_label, gt_partiallabel_save_path)
    print('============Generate GT Sparse Label End============')

    print('============Generate gtpointSparse Label Start============')
    gtpoint_partiallabel_save_path = os.path.join(exp_dir, 'gtpoint_partiallabel')
    generate_sparse_plabel(args.data_dir_target_point, step2_pseudoseg_save_path, gtpoint_partiallabel_save_path)
    print('============Generate gtpointSparse Label End============')

    partial_pseudolabel_save_path = step2_pseudoseg_save_path

    usecuda = True

    makedatalist(args.data_dir_img, args.data_list)
    makedatalist(args.data_dir_target, args.data_list_target)
    makedatalist(args.data_dir_val, args.data_list_val)

    trainloader = data.DataLoader(
        sourceDataSet_train(args.data_dir_img, args.data_dir_label, args.data_list,
                            max_iters=args.num_steps,
                            crop_size=args.input_size,
                            batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(
        kashthuriDataSet_train_step2_entroy(args.data_dir_target, partial_pseudolabel_save_path, args.data_dir_target_point, gtpoint_partiallabel_save_path,
                                  args.data_list_target, max_iters=args.num_steps, iter_start=args.iter_start,
                                  crop_size=args.input_size, batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    targetloader_iter = enumerate(targetloader)

    valloader = data.DataLoader(
        targetDataSet_val(args.data_dir_val, args.data_dir_val_label, args.data_dir_target_val_det, args.data_list_val,
                          crop_size=args.input_size, batch_size=args.batch_size),
        batch_size=1, shuffle=False)

    model = WDA_Net_r(in_channels=1, out_channels=2, device=args.gpu)
    model_label = labelDiscriminator_seg(num_classes=2)
    count_model = Counting_Model(in_channels=1, out_channels=1, device=args.gpu)

    model.train()
    model_label.train()

    if usecuda:
        cudnn.benchmark = True
        cudnn.enabled = True
        model.cuda(args.gpu)
        model_label.cuda(args.gpu)
        count_model.cuda(args.gpu)

    print('load model from:', model_path)
    model.load_state_dict(torch.load(model_path))
    if Dmodel_path != None:
        print('load Dmodel from:', Dmodel_path)
        model_label.load_state_dict(torch.load(Dmodel_path))
    else:
        print('Pretrain Dmodel is None !!!')

    print('restore pretrain counting model from:', args.countingmodel_path)
    count_model.load_state_dict(torch.load(args.countingmodel_path, map_location="cuda:" + str(args.gpu)))

    for param in count_model.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)

    optimizer_label = optim.Adam(model_label.parameters(), lr=args.learning_rate_Dl, betas=(0.9, 0.99))
    scheduler_label = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=args.step_size_Dl, gamma=0.9)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    source_label = 1
    target_label = 0

    sce = SCELoss(num_classes=2, a=1, b=0)

    for i_iter in range(args.iter_start, args.num_steps+1):

        loss_seg_source_value = 0
        loss_seg_target_value = 0
        loss_adv_label_value = 0
        loss_adv_value = 0
        loss_Dlabel_value = 0
        loss_det_source_value = 0
        loss_det_target_value = 0

        optimizer.zero_grad()
        optimizer_label.zero_grad()

        for param in model_label.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        images, labels, points, gaussians, _, _ = batch
        if usecuda:
            images_source = images.cuda(args.gpu)
            points = points.cuda(args.gpu)
        else:
            images_source = images

        seg_source, det_source, _ = model(images_source)

        loss_seg = sce(seg_source, labels, args.gpu, usecuda)
        loss_seg_source_value += loss_seg.data.cpu().numpy()

        loss_det_source = weight_mse_source(det_source, gaussians, args.gpu)
        loss_det_source_value += loss_det_source.data.cpu().numpy()

        loss = loss_seg + 0.1*loss_det_source
        loss.backward()

        _, batch = targetloader_iter.__next__()
        images, tlabels, tpoints, tgaussians, _, _, plabels = batch
        if usecuda:
            images_target = images.cuda(args.gpu)
        else:
            images_target = images

        seg_target, det_target, count_target = model(images_target)

        Dlabel_out = model_label(torch.softmax(seg_target,dim=1))
        adv_source_label = torch.FloatTensor(Dlabel_out.data.size()).fill_(source_label).cuda(args.gpu)
        loss_adv_label = bce_loss(Dlabel_out, adv_source_label)
        loss_adv_value += loss_adv_label.data.cpu().numpy()

        background = (plabels.data.cpu().numpy() == 0)

        loss_seg_target = sce(seg_target, plabels, args.gpu, usecuda)
        loss_seg_target_value += loss_seg_target.data.cpu().numpy()

        target_predict = count_model(images_target.detach())
        loss_ranking = ranking_loss(count_target, target_predict, args.gpu)

        loss_det_target = weight_mse_target(det_target, background, tpoints, tgaussians, args.gpu)
        loss_det_target_value += loss_det_target.data.cpu().numpy()
        loss = 0.001*loss_adv_label + 0.1 * loss_det_target + loss_seg_target + (1 - i_iter/args.num_steps)*loss_ranking
        loss.backward()

        for param in model_label.parameters():
            param.requires_grad = True

        seg_source = seg_source.detach()
        D_out = model_label(torch.softmax(seg_source, dim=1))
        D_source_label = torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda(args.gpu)
        loss_D = bce_loss(D_out, D_source_label)
        loss_D = loss_D / 2
        loss_D.backward()

        seg_target = seg_target.detach()
        D_out = model_label(torch.softmax(seg_target, dim=1))
        D_target_label = torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda(args.gpu)
        loss_D = bce_loss(D_out, D_target_label)
        loss_D = loss_D / 2
        loss_D.backward()

        optimizer.step()
        optimizer_label.step()

        if scheduler is not None:
            scheduler.step()
            args.learning_rate = scheduler.get_last_lr()[0]
        if scheduler_label is not None:
            scheduler_label.step()
            args.learning_rate_Dl = scheduler_label.get_last_lr()[0]

        if (i_iter % 50 == 0):
            print(exp_dir.split('/')[-1] + '_time = {0},lr = {1: 5f}'.format(datetime.datetime.now(),
                                                   args.learning_rate))
            print(
                'iter = {0:8d}/{1:8d}, loss_seg = {2:.5f} loss_adv1 = {3:.5f}, loss_Dlabel = {4:.5f},'
                'loss_det_source = {5:.5f} loss_det_target = {6:.5f} loss_seg_target_value = {7:.5f}'.format(
                    i_iter, args.num_steps, loss_seg_source_value, loss_adv_label_value,
                    loss_Dlabel_value, loss_det_source_value, loss_det_target_value,
                    loss_seg_target_value))


        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            seg_val_dir = os.path.join(exp_dir, 'val', 'cvlab', 'iter_' + str(i_iter), 'seg')
            det_val_dir = os.path.join(exp_dir, 'val', 'cvlab', 'iter_' + str(i_iter), 'det')
            make_dirs(seg_val_dir)
            make_dirs(det_val_dir)
            dice, jac = validate_model(model, valloader, seg_val_dir, det_val_dir, args.gpu, usecuda, type='cvlab')
            print('val dice: %4f' % dice, 'val jac: %4f' % jac)
            if jac > args.best_tjac:
                args.best_tjac = jac
                print('best val dice: %4f' % dice, 'best val jac: %f' % jac)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best.pth'))
                torch.save(model_label.state_dict(),
                           osp.join(args.snapshot_dir, 'bestD.pth'))


if __name__ == '__main__':

    exp_root_dir = 'WDA_VNC2CVLAB/'
    exp_base_name = 'vnc2cvlab_stage3'
    last_exp_dir = 'WDA_VNC2CVLAB/vnc2cvlab_stage1'
    exp_base_dir = exp_root_dir + exp_base_name

    
    for round in range(1, 2):
        args.learning_rate = 0.00005

        exp_dir = exp_base_dir
        remove_or_create_exp_dir(exp_dir)
        last_exp_model_dir = last_exp_dir + '/snapshots'
        print('last_exp_model_dir: ', last_exp_model_dir)

        last_model_path = os.path.join(last_exp_model_dir, 'best.pth')
        last_Dmodel_path = os.path.join(last_exp_model_dir, 'bestD.pth')

        step2_pseudoseg_save_path = exp_dir + '/pseudolab'
        step2_pseudodet_save_path = exp_dir + '/pseudodet'
        print('============Generate Pseudo Label Start============')
        val_main_entroy(last_model_path, step2_pseudoseg_save_path, step2_pseudodet_save_path, args)
        print('============Generate Pseudo Label End============')
        main(exp_dir, last_model_path, last_Dmodel_path, step2_pseudoseg_save_path)
        last_exp_dir = exp_dir


