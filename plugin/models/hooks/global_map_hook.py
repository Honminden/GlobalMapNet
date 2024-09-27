from torch.nn.parallel import DistributedDataParallel
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class GlobalMapHook(Hook):
    """Clear Global Map memory after certain training iteration or at the last training iteration. 
    Turn off global map memory before certain training iteration.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
        start_keep_global_map_iter (int): Start keeping global map after certain training iteration.
    """

    def __init__(self, interval=-1, eval_interval=-1, start_keep_global_map_iter=-1):
        self.interval = interval
        self.eval_interval = eval_interval
        self.start_keep_global_map_iter = start_keep_global_map_iter

    def before_train_iter(self, runner):
        if self.start_keep_global_map_iter < 0:
            return
        if runner.iter == 0:
            model = runner.model
            if isinstance(model, MMDistributedDataParallel) or isinstance(model, DistributedDataParallel):
                model = model.module
            model.keep_global_map = False

        if runner.iter == self.start_keep_global_map_iter:
            model = runner.model
            if isinstance(model, MMDistributedDataParallel) or isinstance(model, DistributedDataParallel):
                model = model.module
            model.keep_global_map = True

        if self.eval_interval <= 0:
            return
        
        if runner.iter == 0 or (runner.iter > 1 and runner.iter % self.eval_interval == 0):
            print("!DEBUG: clear global map memory in before_train_iter")
            model = runner.model
            if isinstance(model, MMDistributedDataParallel) or isinstance(model, DistributedDataParallel):
                model = model.module
            if model.map_builder != None:
                model.init_global_args()

    def after_train_iter(self, runner):
        if self.interval <= 0:
            return
        if self.every_n_iters(runner, self.interval) or self.is_last_iter(runner):
            print("!DEBUG: clear global map memory in after_train_iter")
            model = runner.model
            if isinstance(model, MMDistributedDataParallel) or isinstance(model, DistributedDataParallel):
                model = model.module
            if model.map_builder != None:
                model.init_global_args()
