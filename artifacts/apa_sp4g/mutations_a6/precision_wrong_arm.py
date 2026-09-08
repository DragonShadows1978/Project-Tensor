"""Prior art: June Gemma/SP4G A2/A4 (2026) precision ablation and
post-block residual capture; Higham/Mary (2022) mixed precision context.
https://doi.org/10.1017/S0962492922000022 . No arithmetic changes; A32
native fp32 attention returns bf16 before o_proj exactly as A4 PrecisionModel.
"""
from apa_sp4g_a4_model import PrecisionModel, ResidualModel, LAYERS
from apa_sp4g_a2_model import PPLCapture
from apa_sp4g_common import A, Red


class A32ResidualModel(PrecisionModel):
    def __init__(self, cell):
        if cell['arm'] != 'A32':
            raise Red('A6_A32_REQUIRED')
        super().__init__(dict(cell, arm='D32'))
        self.directory = A/'propagation_a6'/cell['id']
        self.residuals = {str(l): [] for l in LAYERS}
        self.residuals['final_norm'] = []
        self.original_block = self.gemma.Gemma4BlockTC.__call__
        self.original_norm = self.model.norm
        self.block_ids = {id(block): i for i, block in enumerate(self.model.layers)}
        self.norm_offset = None
        owner = self

        def block(obj, x, ropes, position_offset=0, cache=None):
            h, new_cache = owner.original_block(obj, x, ropes, position_offset, cache)
            idx = owner.block_ids[id(obj)]
            if idx in LAYERS:
                owner.save_residual(str(idx), h, position_offset)
            if idx == 47:
                owner.norm_offset = position_offset
            return h, new_cache

        class NormCapture:
            def __call__(self, h):
                out = owner.original_norm(h)
                if owner.norm_offset is None:
                    raise Red('A6_FINAL_NORM_WITHOUT_LAYER47')
                owner.save_residual('final_norm', owner.gemma._cast(out), owner.norm_offset)
                owner.norm_offset = None
                return out

        self.gemma.Gemma4BlockTC.__call__ = block
        self.model.norm = NormCapture()

    save_residual = ResidualModel.save_residual
    finish_residual = ResidualModel.finish_residual

    def close(self):
        self.gemma.Gemma4BlockTC.__call__ = self.original_block
        self.model.norm = self.original_norm
        super().close()


class CaptureModel(PPLCapture):
    def __init__(self, cell, delta):
        super().__init__(cell, delta)
        self.directory = A/'captures_a6'/cell['id']
