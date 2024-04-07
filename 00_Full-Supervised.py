from imports import *

args = get_arguments()

exp_root = 'Full_Supervised'
exp_dir = exp_root + '/01_seg'
args.learning_rate = 0.0005

def main():

    remove_or_create_exp_dir(exp_dir)
    args.snapshot_dir = exp_dir + '/snapshots'
    make_dirs(args.snapshot_dir)
    print('============' + exp_dir + '============')
    setup_seed(args.seed)

    sys.stdout = Logger(stream=sys.stdout, filename=exp_dir + '/trainlog.log')
    sys.stderr = Logger(stream=sys.stderr, filename=exp_dir + '/trainlog.log')

    usecuda = True

    args.data_dir_img = './data/50%vncdata/train/img'
    args.data_dir_label = './data/50%vncdata/train/lab'

    args.data_dir_val = './data/50%vncdata/test/img'
    args.data_dir_val_label = './data/50%vncdata/test/lab'
    args.data_dir_target_val_det = './data/50%vncdata/test/lab'

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

    valloader = data.DataLoader(
        targetDataSet_val(args.data_dir_val, args.data_dir_val_label, args.data_dir_target_val_det, args.data_list_val,
                          crop_size=args.input_size, batch_size=args.batch_size),
        batch_size=1, shuffle=False)

    model = PGS_Net(in_channels=1, out_channels=2, device=args.gpu)
    model.train()

    if usecuda:
        cudnn.benchmark = True
        cudnn.enabled = True
        model.cuda(args.gpu)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)

    sce = SCELoss(num_classes=2, a=1, b=0)

    for i_iter in range(args.iter_start, args.num_steps+1):

        loss_seg_value = 0
        loss_det_source_value = 0

        optimizer.zero_grad()

        _, batch = trainloader_iter.__next__()
        images, labels, points, gaussians, _, _ = batch
        if usecuda:
            images_source = images.cuda(args.gpu)
            points = points.cuda(args.gpu)
        else:
            images_source = images

        seg_source, det_source, count_source = model(images_source)

        loss_seg = sce(seg_source, labels, args.gpu, usecuda)
        loss_seg_value += loss_seg.data.cpu().numpy()
        loss_det_source = weight_mse_source(det_source, points, gaussians, args.gpu)
        loss_det_source_value += loss_det_source.data.cpu().numpy()
        loss_count = consistency_loss(torch.from_numpy(count_source).cuda(args.gpu).unsqueeze(0), gt_num)
        loss_count_value += loss_count.data.cpu().numpy()
        loss = loss_seg + 0.0001 * loss_det_source + 0.001 * loss_count
        loss.backward()

        label = torch.argmax(seg_source, dim=1).float()
        sdice, sjac = dice_coeff(label.cpu(), labels.cpu())

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            args.learning_rate = scheduler.get_last_lr()[0]

        if (i_iter % 50 == 0):
            print('time = {0},lr = {1: 5f},lr_Dl = {2: 6f}'.format(datetime.datetime.now(),
                                                                   args.learning_rate,
                                                                   args.learning_rate_Dl))
        if (i_iter % 50 == 0):
            print(exp_dir+' time = {0},lr = {1: 5f}'.format(datetime.datetime.now(),
                                                   args.learning_rate))
            print('iter = {0:8d}/{1:8d}, loss_source_seg = {2:.5f} loss_det_source_value = {3:.5f}, '.format(
                    i_iter, args.num_steps, loss_seg_value, loss_det_source_value))
            print(
                'sdice2 = {0:.5f} sjac2 = {1:.5f}'.format(
                    sdice, sjac,))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            seg_val_dir = os.path.join(exp_dir, 'val/vnc', 'iter_' + str(i_iter), 'seg')
            det_val_dir = os.path.join(exp_dir, 'val/vnc', 'iter_' + str(i_iter), 'det')
            make_dirs(seg_val_dir)
            make_dirs(det_val_dir)
            dice, jac = validate_model(model, valloader, seg_val_dir, det_val_dir, args.gpu, usecuda)
            print('val dice: %4f' % dice, 'val jac: %4f' % jac)
            if jac > args.best_tjac:
                args.best_tjac = jac
                print('best val dice: %4f' % dice, 'best val jac: %f' % jac)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir,  'best.pth'))


if __name__ == '__main__':
    main()
