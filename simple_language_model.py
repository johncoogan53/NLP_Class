import math

# generate english doc
text = "Four score and seven years ago, our fathers brought forth, upon this continent, a new nation"

# tokenize - split document into a list of little strings
tokens = [char for char in text]

# encode as {0,1}
encodings = [0 if token == " " else 1 for token in tokens]


class SimplestModel:
    def __init__(self):
        self.p_space = None
        pass

    def train(self, encodings):
        # assign a value to the parameter
        self.p_space = encodings.count(0) / len(encodings)
        pass

    def apply(self, encodings):
        return sum(
            math.log(self.p_space) if encoding == 0 else math.log(1 - self.p_space)
            for encoding in encodings
        )
        pass


# define model
model = SimplestModel()
print(model.p_space)
model.train(encodings)

# divide by len encoding because that is n0 + n1 and our new model is n0/(n0+n1)


# compute probability
log_p = sum(
    math.log(p_space) if encoding == 0 else math.log(1 - p_space)
    for encoding in encodings
)

# print
print(log_p)
print(math.exp(log_p))
