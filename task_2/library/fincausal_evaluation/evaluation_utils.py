import re
from collections import defaultdict
from funcy import lflatten
from nltk.tokenize import word_tokenize


def s2dict(lines, lot):
    """
    :param lines: list of sentences or words as strings containing at least two nodes to be mapped in dict
    :param lot: list of tags to be mapped in the dictionary as keys
    :return: dict with keys == tag and values == sentences /words
    """
    d = defaultdict(list)
    for line_, tag_ in zip(lines, lot):
        d[tag_] = line_
    return d


def make_causal_input(lod):
    """
    :param lod: list of dictionaries
    :return: dict of list of tuples for each sentence
    """

    dd = defaultdict(list)
    dd_ = []
    rxlist = [r'("\\)', r'(\\")']
    rx = re.compile('|'.join(rxlist))
    for i in range(len(lod)):
        line_ = lod[i]['sentence']
        line = re.sub(rx, '', line_)
        ante = lod[i]['cause']
        ante = re.sub(rx, '', ante)
        cons = lod[i]['effect']
        cons = re.sub(rx, '', cons)

        d = defaultdict(list)
        index = 0
        for idx, w in enumerate(word_tokenize(line)):
            index = line.find(w, index)

            if not index == -1:
                d[idx].append([w, index])

                index += len(w)

        d_ = defaultdict(list)
        for idx in d:

            d_[idx].append([tuple([d[idx][0][0], '_']), d[idx][0][1]])

            init_a = line.find(ante)
            init_c = line.find(cons)

            for el in word_tokenize(ante):
                start = line.find(el, init_a)
                stop = line.find(el, init_a) + len(el)
                word = line[start:stop]
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([word, 'C']), line.find(word, init_a)])
                    d_[idx] = und_[idx]
                init_a += len(word)

            for el in word_tokenize(cons):
                start = line.find(el, init_c)
                stop = line.find(el, init_c) + len(el)
                word = line[start:stop]
                if int(start) == int(d_[idx][0][1]):
                    und_ = defaultdict(list)
                    und_[idx].append([tuple([word, 'E']), line.find(word, init_c)])
                    d_[idx] = und_[idx]
                init_c += len(word)

        dd[i].append(d_)

    for dict_ in dd:
        dd_.append([item[0][0] for sub in [[j for j in i.values()] for i in lflatten(dd[dict_])] for item in sub])

    return dd_
