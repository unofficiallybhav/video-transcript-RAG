"""One paced groq call, shared by the dataset builder and the evaluator.

The free tier allows 8k tokens per minute, which a single large prompt can come close to
on its own, so requests are paced against the rate limit headers rather than retried
blindly after a failure.
"""
import re
import time

TOKEN_BUDGET = 8000
_remaining = {"tokens": TOKEN_BUDGET, "reset": 0.0}


class DailyLimitReached(RuntimeError):
    """The per-day token budget is gone; waiting will not help until it resets.

    This is separate from the per-minute limit because the response headers do not
    report it: groq only mentions tokens-per-day in the body of the 429, and a run that
    retried it would sleep for hours and still fail.
    """


def parse_reset(value):
    """Turns a groq reset header like '2m52.8s' or '1.507s' into seconds."""
    if not value:
        return 0.0
    seconds = 0.0
    minutes = re.search(r"(\d+)m", value)
    if minutes:
        seconds += int(minutes.group(1)) * 60
    rest = re.search(r"([\d.]+)s", value)
    if rest:
        seconds += float(rest.group(1))
    return seconds


def complete(model, prompt=None, messages=None, retries=8):
    """One chat completion, paced to stay inside the per-minute token budget.

    Either a bare prompt or a full message list, so the evaluator can replay the
    deployed path's own system+user messages through the same rate limiting.
    """
    import video_transcript_rag as rag

    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    sized = "".join(m["content"] for m in messages)

    for _attempt in range(retries):
        needed = len(sized) // 4 + 500
        if _remaining["tokens"] < needed:
            wait = _remaining["reset"] + 1
            print(f"  pacing: {_remaining['tokens']} tokens left, need {needed}, "
                  f"waiting {wait:.0f}s")
            time.sleep(wait)
            _remaining["tokens"] = TOKEN_BUDGET

        try:
            raw = rag.get_groq().chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            # the daily budget cannot be waited out inside a run
            if "per day" in str(e) or "TPD" in str(e):
                raise DailyLimitReached(str(e)) from e
            if "rate" not in str(e).lower() and "429" not in str(e):
                raise
            found = re.search(r"try again in ([\dms.]+)", str(e))
            wait = parse_reset(found.group(1)) if found else 20
            print(f"  rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
            _remaining["tokens"] = TOKEN_BUDGET
            continue

        if raw.headers.get("x-ratelimit-remaining-tokens"):
            _remaining["tokens"] = int(raw.headers["x-ratelimit-remaining-tokens"])
            _remaining["reset"] = parse_reset(raw.headers.get("x-ratelimit-reset-tokens"))

        return raw.parse().choices[0].message.content

    raise RuntimeError(f"giving up after {retries} rate-limited attempts")


def extract_json(content):
    """Pulls the first json object or array out of a model reply."""
    match = re.search(r"\{.*\}|\[.*\]", content or "", re.S)
    if not match:
        return None
    import json
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
