def cpy_eval_args_to_config(args):
    config = {}
    config["model"] = args.model
    config["warm"] = args.warm
    config["iters"] = args.iters
    config["dataset"] = args.dataset
    config["mixed_precision"] = args.mixed_precision
    config["lookup"] = {}
    config["lookup"]["pyramid_levels"] = args.lookup_pyramid_levels
    config["lookup"]["radius"] = args.lookup_radius
    config["cuda_corr"] = args.cuda_corr

    return config
