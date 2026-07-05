"""
Signal generator — converts a price prediction into a trading signal.

BUY  : predicted price is > current price by at least SIGNAL_THRESHOLD
        AND ensemble confidence >= CONFIDENCE_THRESHOLD
SELL : predicted price is < current price by at least SIGNAL_THRESHOLD
        AND ensemble confidence >= CONFIDENCE_THRESHOLD
HOLD : everything else
"""


class SignalGenerator:
    """Converts model predictions into BUY / SELL / HOLD signals."""

    def __init__(self, threshold: float = None,
                 confidence_threshold: float = None):
        """
        Args:
            threshold:            Minimum predicted price move (fractional)
                                  to trigger a signal. Default from config.
            confidence_threshold: Minimum ensemble confidence score [0,1]
                                  to act on a signal. Default from config.
        """
        import config
        self.threshold            = threshold if threshold is not None \
                                    else config.SIGNAL_THRESHOLD
        self.confidence_threshold = confidence_threshold \
                                    if confidence_threshold is not None \
                                    else config.CONFIDENCE_THRESHOLD

    def generate_from_return(self, current_price: float,
                             predicted_return: float,
                             confidence: float) -> dict:
        """
        Generate a signal from a predicted NEXT-DAY RETURN (the native
        output of the ensemble, e.g. 0.004 = +0.4%).

        Args:
            current_price:    Most recent close price.
            predicted_return: Model's predicted next-day simple return.
            confidence:       Ensemble confidence score in [0, 1].

        Returns:
            Same dict as generate().
        """
        predicted_price = current_price * (1.0 + predicted_return)
        return self.generate(current_price, predicted_price, confidence)

    def generate(self, current_price: float,
                 predicted_price: float,
                 confidence: float) -> dict:
        """
        Generate a trading signal.

        Args:
            current_price:   Most recent close price.
            predicted_price: Model's predicted next price.
            confidence:      Ensemble confidence score in [0, 1].
                             Use EnsembleModel.get_confidence() or 0.7
                             for single models.

        Returns:
            dict with keys:
                signal      : 'BUY' | 'SELL' | 'HOLD'
                confidence  : float — passed through unchanged
                predicted   : float — predicted price
                current     : float — current price
                delta_pct   : float — (predicted - current) / current
        """
        if current_price <= 0:
            return self._result('HOLD', current_price, predicted_price,
                                confidence, 0.0)

        delta_pct = (predicted_price - current_price) / current_price

        confident_enough = confidence >= self.confidence_threshold

        if delta_pct > self.threshold and confident_enough:
            signal = 'BUY'
        elif delta_pct < -self.threshold and confident_enough:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return self._result(signal, current_price, predicted_price,
                            confidence, delta_pct)

    @staticmethod
    def _result(signal, current, predicted, confidence, delta_pct):
        return {
            'signal':     signal,
            'confidence': round(confidence, 4),
            'predicted':  round(predicted, 4),
            'current':    round(current, 4),
            'delta_pct':  round(delta_pct * 100, 4),  # stored as %
        }


if __name__ == "__main__":
    sg = SignalGenerator(threshold=0.01, confidence_threshold=0.60)

    cases = [
        (150.0, 153.0, 0.75, "expect BUY   — 2% up, high confidence"),
        (150.0, 151.0, 0.75, "expect HOLD  — 0.67% up, below threshold"),
        (150.0, 153.0, 0.50, "expect HOLD  — 2% up, low confidence"),
        (150.0, 147.0, 0.75, "expect SELL  — 2% down, high confidence"),
        (150.0, 150.0, 0.90, "expect HOLD  — no change"),
    ]

    for current, predicted, conf, note in cases:
        result = sg.generate(current, predicted, conf)
        print(f"{note}")
        print(f"  → {result}")
        print()

    print("signals/generator.py: OK")
