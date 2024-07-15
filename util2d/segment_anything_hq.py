from segment_anything import SamPredictor, build_sam, build_sam_hq

class SAM_HQ:
    def __init__(self, cfg):
        # Segment Anything
        self.sam_predictor = SamPredictor(build_sam_hq(checkpoint=cfg.foundation_model.sam_checkpoint).to("cuda"))
        print('------- Loaded Segment Anything HQ -------')