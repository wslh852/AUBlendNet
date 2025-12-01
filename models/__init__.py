def get_model(cfg):
    ## old
    if cfg.arch == 'stage1':
        from models.stage1 import VQAutoEncoder as Model
        model = Model(args=cfg)
    elif cfg.arch == 'stage2':
        from models.stage2 import AUBlendNet as Model
        model = Model(args=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model
