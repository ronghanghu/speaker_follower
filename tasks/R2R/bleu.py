import tempfile
from os.path import join
import subprocess
import re
import sys
import os
import numpy as np

BLEU_PATH = join("scripts", "multi-bleu.perl")

BASE_REF_FNAME = "ref"
HYP_FNAME = "hyp"


def call_bleu(base_ref_fname, hyp_fname):
    command = "perl %s %s < %s" % (BLEU_PATH, base_ref_fname, hyp_fname)
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)\
        .stdout.read().decode("utf-8")

    match = re.match("BLEU = ([\d.]+),.*BP=([\d.]+),.*\)", result)
    if match is not None:
        # print(match.group(0))
        bleu = float(match.group(1))
        brevity_penalty = float(match.group(2))
        unpenalized = 0
        if brevity_penalty != 0:
            unpenalized = bleu / brevity_penalty
        return bleu, unpenalized

    else:
        sys.stderr.write(
            "warning: BLEU score not found in output file, returning 0")
        return 0, 0


def read_file(fname):
    with open(fname) as f:
        return [line.split() for line in f]


def multi_bleu(multiple_references, hypotheses):
    dir = tempfile.mkdtemp()
    num_refs = len(multiple_references[0])
    assert(all(len(l) == num_refs for l in multiple_references))

    base_ref_fname = join(dir, BASE_REF_FNAME)
    for i in range(num_refs):
        ref_fname = base_ref_fname + str(i)
        with open(ref_fname, 'w') as f:
            for refs in multiple_references:
                f.write("%s\n" % ' '.join(refs[i]))

    hyp_fname = join(dir, HYP_FNAME)
    with open(hyp_fname, 'w') as f:
        for hyp in hypotheses:
            f.write("%s\n" % ' '.join(hyp))

    bleu, unpenalized_bleu = call_bleu(base_ref_fname, hyp_fname)

    # clean up
    for i in range(num_refs):
        ref_fname = base_ref_fname + str(i)
        os.remove(ref_fname)

    os.remove(hyp_fname)
    os.rmdir(dir)

    return bleu, unpenalized_bleu


def single_bleu( references, hypotheses):
    return multi_bleu([[ref] for ref in references], hypotheses)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_fname")
    parser.add_argument("hyp_fname")
    parser.add_argument("--sentence_level", action='store_true')
    parser.add_argument("--nltk", action='store_true')
    args = parser.parse_args()

    refs = read_file(args.ref_fname)
    hyps = read_file(args.hyp_fname)

    # f_bleu = call_bleu(args.ref_fname, args.hyp_fname)
    # l_bleu = multi_bleu([[r] for r in refs], hyps)
    # c_bleu = single_bleu(refs, hyps)
    # assert(f_bleu == l_bleu == c_bleu)

    if args.sentence_level:
        scores = []
        for (ref, hyp) in zip(refs, hyps):
            if args.nltk:
                import nltk
                scores.append(
                    nltk.translate.bleu_score.sentence_bleu([ref], hyp))
            else:
                scores.append(single_bleu([ref], [hyp])[0])
        result = np.mean(scores)
    else:
        if args.nltk:
            import nltk
            result = nltk.translate.bleu_score.corpus_bleu(
                [[r] for r in refs], hyps)
        result = single_bleu(refs, hyps)[0]

    print(result)
