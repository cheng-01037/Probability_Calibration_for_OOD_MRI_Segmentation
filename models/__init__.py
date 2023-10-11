"""
Model interface. We might need to simplify this part
"""

def create_lts(opt, *argv):
    print(opt.model)
    from .lts_tuner import SegmenterNet
    model = SegmenterNet()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

def create_proposed(opt, *argv):
    print(opt.model)
    from .proposed_tuner import SegmenterNet
    model = SegmenterNet()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

