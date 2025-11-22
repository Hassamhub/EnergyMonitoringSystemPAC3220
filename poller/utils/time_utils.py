from datetime import datetime, timezone

def utcnow():
    return datetime.now(timezone.utc)

def date_part(dt: datetime):
    return dt.date()

def hour_part(dt: datetime):
    return dt.hour