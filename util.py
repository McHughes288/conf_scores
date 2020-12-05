from bs4 import BeautifulSoup


def parse_sgml_file(file):
    with open(file, "r") as sgml_f:
        data = BeautifulSoup(sgml_f.read())
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


def parse_ctm_file(file):
    pass
