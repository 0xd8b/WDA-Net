from imports import *
from model.HSC82 import WDA_Net_r
from model.discriminator import labelDiscriminator_seg

args = get_arguments()

args.exp_root_dir = 'WDA_VNC2CVLAB44/'
exp_name = 'vnc2cvlab_stage1'
exp_dir = args.exp_root_dir + exp_name
remove_or_create_exp_dir(exp_dir)

args.snapshot_dir = exp_dir + '/snapshots'
make_dirs(args.snapshot_dir)

args.restore_from = './pretrain_model/vnc_full_supervised1.pth'
args.countingmodel_path = './pretrain_model/vnc_count.pth'
args.learning_rate = 0.00005
args.num_steps = 100000

def main():

    print('============' + exp_name + '============')

    sys.stdout = Logger(stream=sys.stdout, filename=exp_dir + '/trainlog.log')
    sys.stderr = Logger(stream=sys.stderr, filename=exp_dir + '/trainlog.log')

    tb_writer = SummaryWriter(exp_dir+'/result/{:s}tb_logs'.format(exp_name))

    setup_seed(args.seed)
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
        targetDataSet_train_step1_segadv(args.data_dir_target, args.data_dir_target_point,args.data_list_target,
                                 max_iters=args.num_steps, iter_start=args.iter_start, crop_size=args.input_size,
                                 batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    targetloader_iter = enumerate(targetloader)


    valloader = data.DataLoader(
        targetDataSet_val(args.data_dir_val, args.data_dir_val_label, args.data_dir_target_val_det, args.data_list_val,
                         crop_size=args.input_size, batch_size=args.batch_size),
        batch_size=1, shuffle=False)

    model = WDA_Net_r(in_channels=1, out_channels=2, device=args.gpu)
    model_count = Counting_Model(in_channels=1, out_channels=1, device=args.gpu)
    model_label = labelDiscriminator_seg(num_classes=2)

    model.train()
    model_label.train()

    print('restore pretrain seg model from:', args.restore_from)
    model.load_state_dict(torch.load(args.restore_from, map_location="cuda:" + str(args.gpu)), strict=False)

    print('restore pretrain counting model from:', args.countingmodel_path)
    model_count.load_state_dict(torch.load(args.countingmodel_path, map_location="cuda:" + str(args.gpu)))
    for param in model_count.parameters():
        param.requires_grad = False

    if usecuda:
        cudnn.benchmark = True
        cudnn.enabled = True
        model.cuda(args.gpu)
        model_label.cuda(args.gpu)
        model_count.cuda(args.gpu)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)

    optimizer_label = optim.Adam(model_label.parameters(), lr=args.learning_rate_Dl, betas=(0.9, 0.999))
    scheduler_label = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=args.step_size_Dl, gamma=0.9)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    source_label = 1
    target_label = 0

    for i_iter in range(args.iter_start, args.num_steps+1):

        loss_seg_source_value = 0
        loss_D_value = 0
        loss_adv_value = 0
        loss_consistency_value = 0
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
        loss_seg = loss_calc(seg_source, labels, args.gpu, usecuda)
        loss_seg_source_value += loss_seg.data.cpu().numpy()
        loss_det_source = weight_mse_source_seg(det_source, points, gaussians, args.gpu)
        loss_det_source_value += loss_det_source.data.cpu().numpy()
        loss = loss_seg + 0.01*loss_det_source
        loss.backward()

        _, batch = targetloader_iter.__next__()
        images, tpoints, tgaussians, _ = batch
        if usecuda:
            images_target = images.cuda(args.gpu)
        else:
            images_target = images

        seg_target, det_target, count_target = model(images_target)

        pred_argmax = torch.argmax(seg_target.detach(), dim=1).data.cpu().numpy()
        slabel = F.softmax(seg_target.detach(), dim=1).data[:, 1, :, :].cpu().numpy()
        background = (slabel < args.bg_thresold) * (pred_argmax == 0)

        Dlabel_out = model_label(F.softmax(seg_target, dim=1))
        adv_source_label = torch.FloatTensor(Dlabel_out.data.size()).fill_(source_label).cuda(args.gpu)

        loss_adv_label = bce_loss(Dlabel_out, adv_source_label)
        loss_adv_value += loss_adv_label.data.cpu().numpy()

        loss_det_target = weight_mse_target(det_target, background, tpoints, tgaussians, args.gpu)
        loss_det_target_value += loss_det_target.data.cpu().numpy()

        source_predict = model_count(images_target.detach())
        loss_consistency = ranking_loss(count_target, source_predict, args.gpu)
        loss_consistency_value += loss_consistency.data.cpu().numpy()

        loss = 0.001*loss_adv_label + 0.1*loss_det_target + (1 - i_iter/100000)*loss_consistency
       

        loss.backward()

        for param in model_label.parameters():
            param.requires_grad = True

        seg_source = seg_source.detach()
        D_out = model_label(F.softmax(seg_source, dim=1))
        D_source_label = torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda(args.gpu)
        loss_D = bce_loss(D_out, D_source_label)
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value += loss_D.data.cpu().numpy()

        seg_target = seg_target.detach()
        D_out = model_label(F.softmax(seg_target, dim=1))
        D_target_label = torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda(args.gpu)
        loss_D = bce_loss(D_out, D_target_label)
        loss_D = loss_D / 2
        loss_D.backward()
        loss_D_value += loss_D.data.cpu().numpy()

        tb_writer.add_scalars('train', {
                                        'loss_seg_source_value': loss_seg_source_value,
                                        'loss_det_target_value': loss_det_target_value,
                                        'loss_det_source_value': loss_det_source_value,
                                        'loss_consistency_value': loss_consistency_value}, i_iter)

        optimizer.step()
        optimizer_label.step()

        if scheduler is not None:
            scheduler.step()
            args.learning_rate = scheduler.get_last_lr()[0]
        if scheduler_label is not None:
            scheduler_label.step()
            args.learning_rate_Dl = scheduler_label.get_last_lr()[0]

        if (i_iter % 50 == 0):
            print('time = {0},lr = {1: 5f},lr_Dl = {2: 6f}'.format(datetime.datetime.now(),
                                                                   args.learning_rate,
                                                                   args.learning_rate_Dl))

        if (i_iter % 50 == 0):
            print(exp_name + '_time = {0},lr = {1: 5f}'.format(datetime.datetime.now(),
                                                              args.learning_rate))
            print(
                'iter = {0:8d}/{1:8d}, loss_source_seg = {2:.5f} loss_det_source_value = {3:.5f}, loss_consistency_value = {4:.5f},'
                'loss_det_target = {5:.5f}, loss_adv_value = {6:.5f}, '.format(
                    i_iter, args.num_steps, loss_seg_source_value, loss_det_source_value,
                    loss_consistency_value, loss_det_target_value, loss_adv_value))

        if i_iter % args.save_pred_every == 0:
            seg_val_dir = os.path.join(exp_dir, 'val/cvlab', 'iter_' + str(i_iter), 'seg')
            det_val_dir = os.path.join(exp_dir, 'val/cvlab', 'iter_' + str(i_iter), 'det')
            make_dirs(seg_val_dir)
            make_dirs(det_val_dir)
            dice, jac = validate_model(model, valloader, seg_val_dir, det_val_dir, args.gpu, usecuda, type='cvlab')
            
            tb_writer.add_scalars('val', {'dice': dice, 'jac': jac}, i_iter)
            print('val dice: %4f' % dice, 'val jac: %4f' % jac)
            if jac > args.best_tjac:
                args.best_tjac = jac
                print('best val dice: %4f' % dice, 'best val jac: %f' % jac)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'best.pth'))
                torch.save(model_label.state_dict(),
                           osp.join(args.snapshot_dir, 'bestD.pth'))



if __name__ == '__main__':
    main()
