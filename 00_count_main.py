from imports import *

args = get_arguments()
import config

exp_dir = '0count/0vnc2cvlab_count'

remove_or_create_exp_dir(exp_dir)
args.snapshot_dir = exp_dir + '/snapshots'
make_dirs(args.snapshot_dir)
args.learning_rate = 0.001
args.restore_from = 'pretrain_model/Segmentation_train-with-source-data.pth'

args.num_steps = 50000
args.save_pred_every = 1000
args.step_size = 2000

args.data_dir_img = './data/orivncdata/img'
args.data_dir_label = './data/orivncdata/lab'
args.data_list = './dataset/vncdata_list/train.txt'

args.data_dir_val = './data/orivncdata/img'
args.data_dir_val_label = './data/orivncdata/lab'
args.data_list_val = './dataset/vncdata_list/test.txt'

cofficient = standar_gaussian(config.get_value())
print(cofficient)
s_best_mae = 10

def main():

    print('============' + exp_dir + '============')

    sys.stdout = Logger(stream=sys.stdout, filename=exp_dir + '/trainlog.log')
    sys.stderr = Logger(stream=sys.stderr, filename=exp_dir + '/trainlog.log')

    tb_writer = SummaryWriter(exp_dir+'/result/{:s}tb_logs'.format(exp_dir))

    setup_seed(args.seed)
    usecuda = True

    # makedatalist(args.data_dir_img, args.data_list)
    # makedatalist(args.data_dir_val, args.data_list_val)

    trainloader = data.DataLoader(
        sourceDataSet_count(args.data_dir_img, args.data_dir_label, args.data_list,
                            max_iters=args.num_steps,
                            crop_size=args.input_size,
                            batch_size=args.batch_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    trainloader_iter = enumerate(trainloader)

    valloader = data.DataLoader(
        sourceDataSet_val_count(args.data_dir_val, args.data_dir_val_label, args.data_list_val, batch_size=args.batch_size),
        batch_size=1, shuffle=False)

    model = Counting_Model(in_channels=1, out_channels=1, device=args.gpu)
    model.load_state_dict(torch.load(args.restore_from, map_location="cuda:" + str(args.gpu)), strict=False)
    model.train()

    if usecuda:
        cudnn.benchmark = True
        cudnn.enabled = True
        model.cuda(args.gpu)

    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)

    for i_iter in range(args.iter_start, args.num_steps + 1):

        loss_value = 0
        loss_det_value = 0
        loss_count_value = 0
        optimizer.zero_grad()
        _, batch = trainloader_iter.__next__()

        images, _, _, source_gt_num, densitymap = batch

        if usecuda:
            images_source = images.cuda(args.gpu)
            gtnum = source_gt_num.cuda(args.gpu)
            densitymap = densitymap.cuda(args.gpu)
        else:
            images_source = images
            gtnum = source_gt_num

        output = model(images_source)

        intergral = torch.sum(output, dim=(2, 3)) / cofficient
        prednum_numpy = intergral.data[0][0].cpu().numpy()
        gtnum_numpy = gtnum.data[0][0].cpu().numpy()

        loss1 = consistency_loss(intergral, gtnum)
        loss2 = consistency_loss(output.squeeze(0), densitymap)
        # loss2 = relative_loss(intergral, gtnum)

        # loss = loss1 + 100*i_iter/args.num_steps*loss2
        loss = loss1 + 0.01*loss2
        loss_det_value += loss2.data.cpu().numpy()
        loss_count_value += loss1.data.cpu().numpy()

        loss_value += loss.data.cpu().numpy()

        tb_writer.add_scalars('train', {
            'loss_value': loss_value,
            'pred': prednum_numpy,
            'gt': gtnum_numpy}, i_iter)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            args.learning_rate = scheduler.get_last_lr()[0]

        if (i_iter % 50 == 0 and i_iter != 0):
            print('time = {0},lr = {1: 5f}'.format(datetime.datetime.now(), args.learning_rate))
            print(
                'iter = {0:8d}/{1:8d}, loss_det = {2:.5f}, loss_count = {3:.5f}, gt_num = {4:.5f},'
                'pred_num = {5:.5f} '.format(i_iter, args.num_steps, loss_det_value, loss_count_value, gtnum_numpy, prednum_numpy))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            val_counting_error = val_count_model(model, valloader, args.gpu, usecuda)


            print('val countingerror: %4f' % val_counting_error)
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'Count%5d.pth'%i_iter))
           
if __name__ == '__main__':
    main()
