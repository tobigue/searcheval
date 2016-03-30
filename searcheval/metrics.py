from __future__ import division
from math import log
from searcheval.helpers import cumsum, divide


try:
    from itertools import izip as zip
    range = xrange
except NameError:
    pass


def mean(vector):
    """
    Calculates the arithmetic mean of the given vector.

    Args:
    -----
        vector : list
            A non-empty list/array of numbers to be averaged.

    Returns:
    --------
        mean : float
            The arithmetic mean of the given vector.
    """
    return sum(vector) / len(vector)


def precision(relevance_vector):
    """
    Calculates the precision of the given relevance vector.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.

    Returns:
    --------
        precision : float
            The precision of the given relevance vector.
    """
    return relevance_vector.count(1) / len(relevance_vector)


def precision_at_rank(relevance_vector, rank):
    """
    Calculates the precision up to a certain rank in the given
    relevance vector.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.
        rank : int
            The rank [1..len(relevance_vector)] up to which the
            precision gets calculated.

    Returns:
    --------
        precision : float
            The precision of the given relevance vector up to rank.

    Raises:
    -------
        IndexError : if rank is out of the vector's range.
    """
    if rank > len(relevance_vector):
        raise IndexError("Rank out of range!")
    return precision(relevance_vector[:rank])


def precision_vector(relevance_vector):
    """
    Calculates the precision for each rank in a given relevance
    vector up to that rank.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.

    Returns:
    --------
        precision_vector : list
            The precision vector for the given relevance vector.
    """
    cumulated_relevance_vector = cumsum(relevance_vector)
    rank_vector = range(1, len(relevance_vector) + 1)
    return divide(cumulated_relevance_vector, rank_vector)


def avg_prec(relevance_vector):
    """
    Calculates the average precision of the given relevance vector.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.

    Returns:
    --------
        avg_precision : float
            The average precision of the given relevance vector.
    """
    P_vec = precision_vector(relevance_vector)
    P_sum = sum([p * r for (p, r) in zip(P_vec, relevance_vector)])
    return P_sum / len(relevance_vector)


def r_prec(relevance_vector, recall_base):
    """
    Calculates the Rprec of the given relevance vector. That is the
    precision at the rank of the recall base (the number of relevant
    documents). Note that at this point precision equals recall.
    If the recall base is larger than the number of retrieved documents,
    all "missing" documents up to the recall base are considered
    to be non-relevant.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.
        recall_base : int
            The number of relevant documents for the query in
            the gold standard.

    Returns:
    --------
        Rprec : float
            Precision at R-th position in the relevance vector for a
            query that has R relevant documents (the recall base).
    """
    if recall_base > len(relevance_vector):
        no_missing = recall_base - len(relevance_vector)
        relevance_vector = relevance_vector + ([0] * no_missing)
    return precision_at_rank(relevance_vector, recall_base)


def recall(relevance_vector, recall_base):
    """
    Calculates the recall of the given relevance vector.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.
        recall_base : int
            The number of relevant documents for the query in
            the gold standard.

    Returns:
    --------
        recall : float
            The recall of the given relevance vector.
    """
    return sum(relevance_vector) / recall_base


def recall_at_rank(relevance_vector, recall_base, rank):
    """
    Calculates the recall up to a certain rank in the given
    relevance vector.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.
        recall_base : int
            The number of relevant documents for the query in
            the gold standard.
        rank : int
            The rank [1..len(relevance_vector)] up to which the
            recall gets calculated.

    Returns:
    --------
        recall : float
            The recall of the given relevance vector up to rank.

    Raises:
    -------
        IndexError : if rank is out of the vector's range.
    """
    if rank > len(relevance_vector):
        raise IndexError("Rank out of range!")
    return recall(relevance_vector[:rank], recall_base)


def recall_vector(relevance_vector, recall_base):
    """
    Calculates the recall for each rank in a given relevance
    vector up to that rank.

    Args:
    -----
        relevance_vector : list
            A non-empty list of relevance values (0 or 1) for
            each rank of a query result.
        recall_base : int
            The number of relevant documents for the query in
            the gold standard.

    Returns:
    --------
        recall_vector : list
            The recall vector for the given relevance vector.
    """
    return [cr / recall_base for cr in cumsum(relevance_vector)]


def nDCG(gain_vector, ideal_gain_vector):
    """
    Calculates the nDCG (normalized discounted cumulative gain)
    of the given gain vector. Note that nDCG can handle graded
    relevance judgements (as opposed to precision/recall which
    are only definded for binary 0/1 relevance). In case of graded
    relevance judgements note that the most relevant documents
    have the highest gain value. This implementation uses the
    log_2(rank+1) variant of DCG.

    Args:
    -----
        gain_vector : list
            A non-empty list of gain values (int) for each
            rank of a query result.
        ideal_gain_vector : list
            A list of ideal gain values for the query (a list of
            (graded) relevance judgements in decreasing order.)

    Returns:
    --------
        nDCG : float
            The nDCG value of the last position of the given gain vector.
    """
    return nDCG_vector(gain_vector, ideal_gain_vector)[-1]


def nDCG_at_rank(gain_vector, ideal_gain_vector, rank):
    """
    Calculates the nDCG (normalized discounted cumulative gain) of the
    given gain vectorat the given rank. This implementation uses the
    log_2(rank+1) variant of DCG.

    Args:
    -----
        gain_vector : list
            A non-empty list of gain values (int) for each
            rank of a query result.
        ideal_gain_vector : list
            A list of ideal gain values for the query (a list of
            (graded) relevance judgements in decreasing order.)
        rank : int
            The rank [1..len(gain_vector)] for which the
            nDCG gets calculated.

    Returns:
    --------
        nDCG : float
            The nDCG value of the given rank of the given gain vector.
    """
    return nDCG_vector(gain_vector, ideal_gain_vector)[rank - 1]


def nDCG_vector(gain_vector, ideal_gain_vector):
    """
    Calculates the nDCG (normalized discounted cumulative gain)
    for every position of the given gain vector. This implementation
    uses the log_2(rank+1) variant of DCG.

    Args:
    -----
        gain_vector : list
            A non-empty list of gain values (int) for each
            rank of a query result.
        ideal_gain_vector : list
            A list of ideal gain values for the query (a list of
            (graded) relevance judgements in decreasing order.)

    Returns:
    --------
        nDCG_vector : list
            The nDCG value for every position of the given gain vector.
    """
    # fit ideal gain vector in size
    I = [0] * len(gain_vector)
    fitted_ideal_gain_vector = ideal_gain_vector[0:len(gain_vector)]
    I[0:len(fitted_ideal_gain_vector)] = fitted_ideal_gain_vector
    # make log vector (uses log_2(rank+1) version of DCG)
    log_vec = [log(x, 2) for x in range(2, len(gain_vector) + 2)]
    # discount
    DG = divide(gain_vector, log_vec)
    DI = divide(I, log_vec)
    # cumulate
    DCG = cumsum(DG)
    DCI = cumsum(DI)
    # finish
    nDCG = divide(DCG, DCI)
    return nDCG
