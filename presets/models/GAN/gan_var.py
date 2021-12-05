class GanVar:
    def __init__(self, d, g):
        self.d = d
        self.g = g

    def __str__(self):
        return f"gan_var({str(self.d)}|  {str(self.g)})"