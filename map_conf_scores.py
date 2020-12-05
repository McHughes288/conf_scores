from util import parse_sgml_file, parse_ctm_file, parse_word_string
from absl import logging, flags, app

flags.DEFINE_string("dev_set", None, "path to development SGML file used to calibrate conf scores")
flags.DEFINE_string("test_set", None, "path to held-out test set to apply calibration to")

flags.mark_flag_as_required("dev_set")
flags.mark_flag_as_required("test_set")

FLAGS = flags.FLAGS


def main(unused_argv):

    test_data = parse_ctm_file(FLAGS.test_set)
    print(test_data[0:10])

    dev_data = parse_sgml_file(FLAGS.dev_set)
    print(dev_data[0:10])


if __name__ == "__main__":
    app.run(main)
