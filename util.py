from bs4 import BeautifulSoup


def parse_sgml_file(file_path):
    with open(file_path, "r") as sgml_f:
        data = BeautifulSoup(sgml_f.read(), "html.parser")
    return data


def parse_word_string(text):
    words = text.split(":")
    word_data = []
    for word in words:
        state, ref, pred, time_bracket, conf = word.split(",")
        if state != "D":
            start, end = time_bracket.split("+")
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
                    "start": start,
                    "end": end,
                    "word": word,
                    "conf": conf,
                }
            )
    return data
