from util import parse_sgml_file, parse_ctm_file, parse_word_string
from absl import logging, flags, app

flags.DEFINE_string("dev_set", None, "path to development SGML file used to calibrate conf scores")
flags.DEFINE_string("test_set", None, "path to held-out test set to apply calibration to")

flags.mark_flag_as_required("dev_set")
# flags.mark_flag_as_required("test_set")

FLAGS = flags.FLAGS


def main(unused_argv):

    data = parse_sgml_file(FLAGS.dev_set)
    relevant_data = []
    for i, segment in enumerate(data("path")):
        print(i, segment, segment.text)

        text = segment.text.strip()

        if text:
            word_data = parse_word_string(text)
            assert int(segment["word_cnt"]) == len(
                word_data
            ), f"Got {len(word_data)} words, expected {segment['word_cnt']}"

            relevant_data.append(
                {
                    "file": segment["file"],
                    "start_time": segment["r_t1"],
                    "end_time": segment["r_t2"],
                    "data": word_data,
                }
            )
        if i > 20:
            break
    print(relevant_data)


if __name__ == "__main__":
    app.run(main)
