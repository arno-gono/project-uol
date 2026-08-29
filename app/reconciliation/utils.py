


def calc_score_agent(
        nb_errors_injected: int,
        nb_errors_found: int,
        nb_false_positive_high: int,
        nb_false_positive_critical: int
) -> float:

    """
        Score system based on the number of anomalies found. If the agent flags more than the anomalies,
        it would just be a check by a human agent in a production environment. But missing an anomaly is where
        there could be some operational consequences.
        A run where nothing was injected has nothing to miss, so it scores full marks for now.
        Decreasing the score of false positive by 0.05 if the severity is marked as High and 0.1 if the severity
        is marked as Critical. The agent cannot be fostered to generate errors to score high, all high and critical
        errors that are false positive decrease the score.
        This function computes on reconciliation_logs results: it can be refined down the line.
    """

    decrease_high = 0.05
    decrease_critical = 0.1

    score_agent = round(nb_errors_found / nb_errors_injected, 4) if nb_errors_injected != 0 else 1
    score_agent -= (nb_false_positive_high * decrease_high + nb_false_positive_critical * decrease_critical)

    return score_agent


if __name__ == "__main__":
    print(calc_score_agent(
        nb_errors_injected=15,
        nb_errors_found=10,
        nb_false_positive_high=0,
        nb_false_positive_critical=1
    ))
