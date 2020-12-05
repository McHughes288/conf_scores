from util import parse_sgml_file, parse_ctm_file, parse_word_string
from absl import logging, flags, app
import numpy as np
from math import isnan

flags.DEFINE_string("dev_set", None, "path to development SGML file used to calibrate conf scores")
flags.DEFINE_string("test_set", None, "path to held-out test set to apply calibration to")
flags.DEFINE_float("lr", 0.1, "path to development SGML file used to calibrate conf scores")
flags.DEFINE_integer("steps", 1000, "path to held-out test set to apply calibration to")

flags.mark_flag_as_required("dev_set")
flags.mark_flag_as_required("test_set")

FLAGS = flags.FLAGS


def piecewise_linear_mapping(x, params):
    m1, m2, m3, c2 = params
    a1 = c2 / (m1 - m2)
    a2 = (1 - m3 - c2) / (m2 - m3)

    f1 = lambda x: m1 * x
    f2 = lambda x: m2 * x + c2
    f3 = lambda x: m3 * (x - 1) + 1

    y = np.piecewise(
        x,
        [(0 <= x) * (x <= a1), (a1 < x) * (x < a2), (a2 <= x) * (x <= 1)],
        [f1, f2, f3],
    )
    return y


def get_loss(confs, labels, params, epsilon=1e-3):
    mapped_confs = piecewise_linear_mapping(confs, params)
    loss = labels * np.log(mapped_confs + epsilon) + (1 - labels) * np.log(
        1 - mapped_confs + epsilon
    )
    return loss.mean()


def get_gradients(confs, labels, params):
    m1, m2, m3, c2 = params
    a1 = c2 / (m1 - m2)
    a2 = (1 - m3 - c2) / (m2 - m3)

    # param grads of first linear segment
    mask = ((0 <= confs) * (confs <= a1)).astype(int)
    y = labels * mask
    x = confs * mask
    grad_m1 = (y / m1) + (1 - y) * (-x / (1 - m1 * x))
    grad_m1 = grad_m1.mean()

    # param grads of second linear segment
    mask = ((a1 < confs) * (confs < a2)).astype(int)
    y = labels * mask
    x = confs * mask
    grad_c2 = (y / (m2 * x + c2)) + (1 - y) * (-1 / (1 - m2 * x - c2))
    grad_c2 = grad_c2.mean()
    grad_m2 = x * grad_c2
    grad_m2 = grad_m2.mean()

    # param grads of third linear segment
    mask = ((a2 <= confs) * (confs <= 1)).astype(int)
    y = labels * mask
    x = confs * mask
    grad_m3 = (y * (x - 1) / (m3 * (x - 1) + 1)) + (1 - y) / m3
    grad_m3 = grad_m3.mean()

    return np.array([grad_m1, grad_m2, grad_m3, grad_c2])


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
    # parse the dev data and extract confidence scores and labels
    dev_data = parse_sgml_file(FLAGS.dev_set)
    confs, labels = extract_train_samples(dev_data)

    # initialise params for piecewise linear mapping gradients m1, m2, m3 and intercept c2
    params = np.array([0.55, 5, 0.55, -2])
    prev_loss = -1000

    # training loop with gradient ascent update
    # result is a mapping that maximises confidence of correctly predicted words
    # and minimises confidence of incorrectly predicted words
    for step in range(FLAGS.steps):
        loss = get_loss(confs, labels, params)
        grads = get_gradients(confs, labels, params)

        delta_loss = loss - prev_loss
        prev_loss = loss
        if step % 20 == 0:
            print(
                f"step {step}, loss {loss}, delta_loss {delta_loss}, params {params}, grads {grads}"
            )

        if isnan(loss):
            print("Loss is nan so breaking training loop")
            break

        if np.abs(delta_loss) < 1e-10:
            print(f"Change in loss is {delta_loss} < 1e-10, so breaking training loop")
            break

        # gradient ascent
        params = params + FLAGS.lr * grads
        # ensure gradient of lines are positive
        m1, m2, m3, c2 = params
        assert m1 > 0
        assert m2 > 0
        assert m3 > 0

    print(f"final loss {loss}, final params {params}")

    # Parse the test data, extract confidence scores and pass through the learnt linear mapping
    test_data = parse_ctm_file(FLAGS.test_set)
    test_confs = np.array([item["conf"] for item in test_data])
    av_conf = test_confs.mean()

    mapped_confs = piecewise_linear_mapping(test_confs, params)
    av_mapped_conf = mapped_confs.mean()

    print(f"av_conf {av_conf}, av_mapped_conf {av_mapped_conf}")


if __name__ == "__main__":
    app.run(main)
