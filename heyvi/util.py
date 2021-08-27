from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo  # python >=3.9
    tz = ZoneInfo('US/Eastern')
except:
    tz = timezone.utc


def timestamp():
    """Datetime stamp in eastern timezone with microsecond resolution"""
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S.%f%z")

