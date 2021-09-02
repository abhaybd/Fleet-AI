from tensorboardX import SummaryWriter

class DummyWriter(object):
    def __init__(self, *_, **__):
        pass

    def __getattr__(self, item):
        if hasattr(SummaryWriter, item):
            return lambda *_, **__: None if callable(getattr(SummaryWriter, item)) else None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
