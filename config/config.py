class Config(object):

    num_classes = 399
    easy_margin = False
    use_se = False

    logs_root = "/content/DmmlTiSV/logs"
    data_dir = "/content/dataset/"

    ims_ids = 64
    ims_per_id = 2
    degree = 45

    use_gpu = True  # use GPU or not

    num_workers = 8  # how many workers for loading data

    max_epoch = 100
    lr = 3e-4  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-3
