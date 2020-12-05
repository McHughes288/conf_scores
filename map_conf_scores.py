"""Map confidence scores

Training script to learn a piece-wise linear mapping that maximises confidence of 
correctly predicted words and minimises confidence of incorrectly predicted words
on a development dataset.
The mapping is then applied to the confidence scores on a held-out dataset.

Example:
    $ python3 -m map_conf_scores --dev_set ~/dev.ctm.sgml --test_set ~/eval.ctm --steps 2000 --lr 0.01
"""

import numpy as np
from math import isnan
from absl import flags, app
from util import parse_sgml_file, parse_ctm_file, parse_word_string

flags.DEFINE_string("dev_set", None, "path to development SGML file used to calibrate conf scores")
flags.DEFINE_string("test_set", None, "path to held-out test set to apply calibration to")
flags.DEFINE_float("lr", 0.1, "learning rate for gradient ascent")
flags.DEFINE_integer("steps", 1000, "number of steps to run the training loop for")

flags.mark_flag_as_required("dev_set")
flags.mark_flag_as_required("test_set")

FLAGS = flags.FLAGS


def piecewise_linear_mapping(x, params):
    """Applies a three stage piecewise linear mapping

    Args:
        x (np.array): array of input values
        params (np.array): array containing four model parameters (m1, m2, m3 and c2)
            These correspond to the slope gradients of each line and the intercept of the second

    Returns:
        y (np.array): array of mapped values

    """
    # unpack params
    m1, m2, m3, c2 = params
    # calculate bounds
    a1 = c2 / (m1 - m2)
    a2 = (1 - m3 - c2) / (m2 - m3)
    # equations of three lines
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
    """Apply the mapping to the confidence scores and calculate the mean log likelihood

    Args:
        confs (np.array): array of confidence scores assumed to be between 0 and 1
        labels (np.array): array of labels assumed to be 0 or 1
        params (np.array): array containing four model parameters (m1, m2, m3 and c2)
        epsilon (float, optional): tiny number to stop log values being undefined

    Returns:
        loss (float): mean of the log likelihood values for each item in the development set

    """
    mapped_confs = piecewise_linear_mapping(confs, params)
    loss = labels * np.log(mapped_confs + epsilon) + (1 - labels) * np.log(
        1 - mapped_confs + epsilon
    )
    return loss.mean()


def get_gradients(confs, labels, params):
    """Calculate the derivative of the loss with respect to each of the model parameters

    Args: identical descriptions to function get_loss

    Returns:
        np.array containing the gradients for m1, m2, m3 and c2

    """
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
    """Extract confidence scores and labels from the parsed dev data

    Args:
        data (list): list of segments containing a dictionary of related data
            (output of parse_sgml_file)

    Returns:
        confs (np.array): array of confidence scores
        labels (np.array): array of labels
    """
    confs, labels = [], []
    for segment in data:
        for word in segment["data"]:
            # If word was predicted to be correct
            if word["state"] == "C":
                label = 1
            # If word results in a substitution or insertion error
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
    for step in range(FLAGS.steps):
        loss = get_loss(confs, labels, params)
        grads = get_gradients(confs, labels, params)
        assert not isnan(loss), "Loss is nan so stopping training"

        # log stats and stop if change in loss drops below the threshold
        delta_loss = loss - prev_loss
        prev_loss = loss
        if step % 20 == 0:
            print(
                f"step {step}, loss {loss}, delta_loss {delta_loss}, params {params}, grads {grads}"
            )
        if np.abs(delta_loss) < 1e-10:
            print(f"Change in loss is {delta_loss} < 1e-10, so breaking training loop")
            break

        # gradient ascent
        params = params + FLAGS.lr * grads
        # ensure slopes of lines are positive
        m1, m2, m3, c2 = params
        assert m1 > 0, f"The slope of line 1 is {m1}, it must be positive"
        assert m2 > 0, f"The slope of line 2 is {m2}, it must be positive"
        assert m3 > 0, f"The slope of line 3 is {m3}, it must be positive"

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
