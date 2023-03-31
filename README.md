# Mapped Confidence Scores

### Main script
Main script is `map_conf_scores.py`

### Description
Training script to learn a piece-wise linear mapping that maximises confidence of 
correctly predicted words and minimises confidence of incorrectly predicted words
on a development dataset.
The mapping is then applied to the confidence scores on a held-out dataset.

### How to run
```
$ python3 -m map_conf_scores \
--dev_set ~/data/dev.ctm.sgml \
--test_set ~/data/eval.ctm \
--out_dir ~/exp \
--steps 2000 \
--lr 0.01 2>&1 | tee ~/results.log
```

### Resources used

SGML parsing
* https://www.crummy.com/software/BeautifulSoup/bs4/doc/
* https://stackoverflow.com/questions/4633162/sgml-parser-in-python

Linear piece-wise mapping for confidence scores suggested in this paper:
http://mi.eng.cam.ac.uk/~ar527/ragni_is2018a.pdf

I implemented maximum likelihood training using gradient ascent update steps. Lectures during my engineering degree at Cambridge allowed me to write down the log likelihood for this problem and differentiate with respect to the model parameters. I used confidence scores as an unsupervised metric in my 4th year project which also helped my understanding.
