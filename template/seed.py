"""
Reproducibility

Doc:
pytorch 
https://pytorch.org/docs/stable/notes/randomness.html

mmengine
https://github.com/open-mmlab/mmengine/blob/9b984056726583f9bafcfb35daad0e4feb97cab9/mmengine/runner/runner.py#L698



"""
# lighnting
# kaggle
# torch
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
