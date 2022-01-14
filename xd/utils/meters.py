class AverageMeter:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.sum += val
        self.count += count
        self.avg = self.sum / self.count
