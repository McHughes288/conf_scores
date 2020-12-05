from util import parse_sgml_file, parse_ctm_file, parse_word_string
from absl import logging, flags, app
import numpy as np

flags.DEFINE_string("dev_set", None, "path to development SGML file used to calibrate conf scores")
flags.DEFINE_string("test_set", None, "path to held-out test set to apply calibration to")

flags.mark_flag_as_required("dev_set")
flags.mark_flag_as_required("test_set")

FLAGS = flags.FLAGS


def piecewise_linear_mapping(x, m1=0.55, m2=5, m3=0.55, c2=-2):
    a1 = c2 / (m1 - m2)
    a2 = (1 - m3 - c2) / (m2 - m3)

    f = np.piecewise(
        x,
        [(0 <= x) * (x <= a1), (a1 < x) * (x < a2), (a2 <= x) * (x <= 1)],
        [lambda x: m1 * x, lambda x: m2 * x + c2, lambda x: m3 * (x - 1) + 1],
    )
    return f


def get_loss(confs, labels, epsilon=1e-5):
    print(len(confs), len(labels))
    print(confs[0:20], labels[0:20])
    mapped_confs = piecewise_linear_mapping(confs)
    print(mapped_confs[0:20])

    loss = labels * np.log(mapped_confs + epsilon) + (1 - labels) * np.log(
        1 - mapped_confs + epsilon
    )
    return loss.mean()


def extract_train_samples(data):
    confs, labels = [], []
    for segment in data:
        for word in segment["data"]:
            if word["state"] == "C":
                label = 1
            elif word["state"] in ["S", "I"]:
                label = 0
            elif word["state"] == "D":
                # ignoring deletions in piece-wise mapping
                continue
            else:
                raise ValueError(f"Did not expect word state {word['state']}")
            confs.append(word["conf"])
            labels.append(label)
    return np.array(confs), np.array(labels)


def main(unused_argv):

    test_data = parse_ctm_file(FLAGS.test_set)
    print(test_data[0:10])

    dev_data = parse_sgml_file(FLAGS.dev_set)
    print(dev_data[0:10])

    confs, labels = extract_train_samples(dev_data)
    loss = get_loss(confs, labels)

    print(loss)


if __name__ == "__main__":
    app.run(main)
