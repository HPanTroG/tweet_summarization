from rouge import Rouge

def compute_rouge_score(can, ref, is_input_files = False):
    """
    Compute ROUGE scores
    :param can: predicted summary
    :param ref: reference summary
    :param is_file (bool): whether the inputs are stored in a file
    :return: dict: ROUGE score
    """
    if is_input_files:
        candidates = [line.strip() for line in open(can, encoding="utf-8")]
        references = [line.strip() for line in open(ref, encoding="utf-8")]
    else:
        candidates = can
        references = ref

    print("#candidates: {}".format(len(candidates)))
    print("#references: {}".format(len(references)))

    evaluator = Rouge()

    scores = evaluator.get_scores(candidates, [[iter] for iter in references])

    return scores