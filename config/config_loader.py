import json


mandatory = {}


class DefaultSetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __setitem__(self, key, value):
        if key not in self.dictionary:
            if value is mandatory:
                raise ValueError(f" Argument --> {key} was mandatory but is not there")
            else:
                self.dictionary[key] = value

    def __getitem__(self, key):
        if key not in self.dictionary:
            self.dictionary[key] = {}
        return DefaultSetter(self.dictionary[key])


def load_json_config(config_path):
    file = open(config_path)
    config = json.load(file)

    default_setter = DefaultSetter(config)
    default_setter["name"] = mandatory
    default_setter["train"]["lr"] = mandatory
    default_setter["train"]["dataset"] = mandatory
    default_setter["train"]["num_steps"] = mandatory
    default_setter["train"]["batch_size"] = mandatory
    default_setter["train"]["image_size"] = mandatory
    default_setter["train"]["validation"] = mandatory
    default_setter["train"]["restore_ckpt"] = None
    default_setter["train"]["iters"] = [4, 6, 8]
    default_setter["train"]["eval_iters"] = mandatory
    default_setter["train"]["loss"] = mandatory
    default_setter["train"]["gamma"] = mandatory
    default_setter["train"]["wdecay"] = mandatory
    default_setter["lr_peak"] = 0.05
    default_setter["mixed_precision"] = True
    default_setter["gpus"] = [0, 1]
    default_setter["epsilon"] = 1e-8
    default_setter["add_noise"] = False
    default_setter["clip"] = 1.0
    default_setter["dropout"] = 0.0
    default_setter["current_phase"] = 0
    default_setter["current_steps"] = -1
    default_setter["fnet_norm"] = "group"
    default_setter["cnet_norm"] = "group"
    default_setter["cuda_corr"] = False

    return config


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
