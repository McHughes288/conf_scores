from bs4 import BeautifulSoup


def parse_sgml_file(file_path):
    with open(file_path, "r") as sgml_f:
        data = BeautifulSoup(sgml_f.read(), "html.parser")

    relevant_data = []
    for i, segment in enumerate(data("path")):
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
