"""Odds conversions and edge calculation."""


def prob_to_decimal(p: float) -> float:
    return 1 / p


def american_to_prob(odds: float) -> float:
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds: float) -> float:
    if odds < 0:
        return 1 + 100 / -odds
    return 1 + odds / 100


def remove_vig(home_implied: float, away_implied: float) -> tuple:
    total = home_implied + away_implied
    return home_implied / total, away_implied / total


def edge(model_prob: float, market_decimal: float) -> float:
    return model_prob * market_decimal - 1
