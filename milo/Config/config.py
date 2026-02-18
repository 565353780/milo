from base_gs_trainer.Config.config import (
    ParamGroup,
    BaseModelParams,
    BasePipelineParams,
    BaseOptimizationParams,
)


class ModelParams(BaseModelParams, ParamGroup):
    def __init__(self, parser, sentinel=False):
        BaseModelParams.__init__(self)

        self._images = "images"

        self.llff = 8
        self.kernel_size = 0.0  # Added

        ParamGroup.__init__(self, parser, "Loading Parameters", sentinel)
        return

class PipelineParams(BasePipelineParams, ParamGroup):
    def __init__(self, parser):
        BasePipelineParams.__init__(self)

        self.depth_ratio = 0.0

        ParamGroup.__init__(self, parser, "Pipeline Parameters")
        return

class OptimizationParams(BaseOptimizationParams, ParamGroup):
    def __init__(self, parser):
        BaseOptimizationParams.__init__(self)

        self.percent_dense = 0.01

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_until_iter = 3000
        self.densify_grad_threshold = 0.0002

        self.random_background = False
        self.appearance_embeddings_lr = 0.001
        self.appearance_network_lr = 0.001

        ParamGroup.__init__(self, parser, "Optimization Parameters")
        return
