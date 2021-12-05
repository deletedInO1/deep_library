class TrainResults:
    def __init__(self, accuracy=0, loss=0, count=0):
        self.raw_accuracy = accuracy
        self.raw_loss = loss
        self.count = count

    def get_accuracy(self):
        return self.raw_accuracy / self.count
    def get_loss(self):
        return self.raw_loss / self.count

    def __add__(self, other):
        assert isinstance(other, TrainResults)
        return TrainResults(self.raw_accuracy + other.raw_accuracy, self.raw_loss + other.raw_loss, self.count + other.count)