"""Rescheduling via update_event must keep all-day semantics and RRULE BYDAY."""
import json
import uuid

import pytest

import core.database as cdb
from core.database import CalendarEvent
from tests.helpers.sqlite_db import make_temp_sqlite

_TS, _ENGINE, _TMPDB = make_temp_sqlite(cdb.Base.metadata)


@pytest.fixture(autouse=True)
def _bind_temp_db(monkeypatch):
    monkeypatch.setattr(cdb, "SessionLocal", _TS)
    import routes.calendar_routes as cr
    monkeypatch.setattr(cr, "SessionLocal", _TS, raising=False)
    yield


async def test_update_all_day_event_keeps_all_day_and_recalcs_end():
    from src.tool_implementations import do_manage_calendar

    owner = "cal-" + uuid.uuid4().hex[:6]
    created = await do_manage_calendar(json.dumps({
        "action": "create_event",
        "summary": "Weekly Trivia",
        "dtstart": "2026-06-26",
        "all_day": True,
        "rrule": "FREQ=WEEKLY;BYDAY=FR",
    }), owner=owner)
    assert created.get("exit_code", 0) == 0, created
    uid = created["uid"]

    updated = await do_manage_calendar(json.dumps({
        "action": "update_event",
        "uid": uid,
        "dtstart": "2026-06-25",
    }), owner=owner)
    assert updated.get("exit_code", 0) == 0, updated

    db = _TS()
    try:
        ev = db.query(CalendarEvent).filter(CalendarEvent.uid == uid).first()
        assert ev.all_day is True
        assert ev.is_utc is False
        assert ev.dtstart.date().isoformat() == "2026-06-25"
        assert ev.dtend.date().isoformat() == "2026-06-26"
        assert "BYDAY=TH" in ev.rrule.upper()
    finally:
        db.close()
