"""
Discord webhook alerter.
Sends notifications on order fills, daily summaries, retrain results, and errors.
All methods fail silently - a broken webhook never crashes a trading job.
"""
import requests
from datetime import datetime, timezone


class DiscordAlerter:
    """Posts messages to a Discord channel via webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def send_order_alert(self, ticker: str, side: str,
                         qty: int, price: float) -> None:
        """Called when an order is submitted."""
        value = qty * price
        self._post(
            f"**ORDER - {side.upper()}** `{ticker}`\n"
            f"> Qty: {qty} shares @ ${price:.2f}\n"
            f"> Value: ${value:,.2f}"
        )

    def send_daily_summary(self, signals: dict,
                           portfolio_value: float,
                           rationale: str = None) -> None:
        """
        Called at the end of every daily job run.

        Args:
            signals:         {ticker: signal_dict} for all tickers.
            portfolio_value: Current total portfolio equity.
            rationale:       Optional plain-English narration of the day's
                             decisions (from monitoring.narrator).
        """
        lines = ["**Daily Summary**",
                 f"> Portfolio: **${portfolio_value:,.2f}**",
                 "> Signals:"]

        for ticker, sig in signals.items():
            action = sig.get('signal', 'N/A')
            delta  = sig.get('delta_pct', 0.0)
            conf   = sig.get('confidence', 0.0)
            lines.append(
                f">   `{ticker}` **{action}**  "
                f"move {delta:+.2f}%  conf {conf:.2f}"
            )

        if rationale:
            lines.append("")
            lines.append(f"_{rationale}_")

        self._post('\n'.join(lines))

    def send_retrain_summary(self, metrics: dict) -> None:
        """
        Called after weekly model retrain.

        Args:
            metrics: {ticker: {'rmse': float, 'dir_acc': float, ...}}
                     RMSE is in next-day return units.
        """
        lines = ["**Weekly Retrain Complete**"]
        for ticker, m in metrics.items():
            rmse    = m.get('rmse', 0)
            dir_acc = m.get('dir_acc', 0)
            lines.append(
                f"> `{ticker}` - RMSE {rmse:.5f} (returns)  "
                f"dir. accuracy {dir_acc:.3f}"
            )
        self._post('\n'.join(lines))

    def send_error(self, error_message: str) -> None:
        """Called on any uncaught exception in a job."""
        self._post(
            f"**ERROR**\n"
            f"> {error_message}\n"
            f"> Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, content: str) -> None:
        """POST message to Discord. Fails silently on any error."""
        if not self.webhook_url:
            return
        try:
            requests.post(
                self.webhook_url,
                json={'content': content},
                timeout=5,
            )
        except Exception:
            pass   # Never crash a job because of a failed alert


if __name__ == "__main__":
    import config

    alerter = DiscordAlerter(config.DISCORD_WEBHOOK_URL)

    print("Sending test alerts to Discord...")

    alerter.send_order_alert("AAPL", "BUY", 77, 194.36)

    alerter.send_daily_summary(
        signals={
            "AAPL":  {'signal': 'BUY',  'delta_pct': 1.45, 'confidence': 0.72},
            "MSFT":  {'signal': 'HOLD', 'delta_pct': 0.30, 'confidence': 0.55},
            "GOOGL": {'signal': 'SELL', 'delta_pct': -1.20, 'confidence': 0.68},
        },
        portfolio_value=101_234.56,
    )

    alerter.send_retrain_summary({
        "AAPL":  {'rmse': 0.68, 'r2': 0.9957},
        "MSFT":  {'rmse': 0.91, 'r2': 0.9941},
        "GOOGL": {'rmse': 1.12, 'r2': 0.9928},
    })

    alerter.send_error("Phase 11 test alert - ignore this message.")

    print("All alerts sent. Check your Discord channel.")
    print("monitoring/alerts.py: OK")
