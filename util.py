from bs4 import BeautifulSoup


def parse_sgml_file(file_path):
    """Parse the HTML style formatting in an SGML file containing segment and word
    prediction information.

    Args:
        file_path (str): path to SGML file

    Returns:
        relevant_data (list): parsed relevant data stored in a dictionary per item

    """
    with open(file_path, "r") as sgml_f:
        data = BeautifulSoup(sgml_f.read(), "html.parser")

    relevant_data = []
    for segment in data("path"):
        text = segment.text.strip()

        # add word information if present, otherwise continue
        if text:
            word_data = parse_word_string(text)
            assert int(segment["word_cnt"]) == len(
                word_data
            ), f"Got {len(word_data)} words, expected {segment['word_cnt']}"

            relevant_data.append(
                {
                    "file": segment["file"],
                    "start": float(segment["r_t1"]),
                    "end": float(segment["r_t2"]),
                    "data": word_data,
                }
            )
    return relevant_data


def parse_word_string(text):
    """Parse the text string containing the word prediction information

    Args:
        text (str): string containing state, reference word, predicted word, times
            and confidence scores

    Returns:
        word_data (list): list of dictionaries, one for each word in the segment

    """
    words = text.split(":")
    word_data = []
    for word in words:
        state, ref, pred, time_bracket, conf = word.split(",")
        # There is no extra information if state is deletion
        if state != "D":
            start, end = time_bracket.split("+")
            start = float(start)
            end = float(end)
            conf = float(conf)
            pred = pred.replace('"', "")

        else:
            pred, start, end, conf = None, None, None, None
        word_data.append(
            {
                "state": state,
                "ref": ref.replace('"', ""),
                "pred": pred,
                "start": start,
                "end": end,
                "conf": conf,
            }
        )
    return word_data


def parse_ctm_file(file_path):
    """Parse the held-out test set data in a CTM format.

    Args:
        file_path (str): path to CTM file

    Returns:
        data (list): parsed data stored in a dictionary per item

    """
    with open(file_path, "r") as ctm_f:
        lines = ctm_f.readlines()
        data = []
        for line in lines:
            try:
                file_name, _, start, end, word, conf = line.split()
            except ValueError:
                continue
            data.append(
                {
                    "file": file_name,
                    "start": float(start),
                    "end": float(end),
                    "word": word,
                    "conf": float(conf),
                }
            )
    return data
