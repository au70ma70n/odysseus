#!/bin/sh
# CalDAV sync fixes on top of rupert648/ferroxide (PR #6):
# - RecurrenceID may be JSON number or string from Proton API
# - QueryCalendarObjects should skip bad events (Odysseus uses date_search)
set -eu
cd /src/ferroxide

python3 <<'PY'
from pathlib import Path

cal = Path("protonmail/calendar.go")
text = cal.read_text()
old = "\tRecurrenceID               string\n"
new = "\tRecurrenceID               interface{} `json:\"RecurrenceID,omitempty\"`\n"
if old not in text:
    raise SystemExit("calendar.go RecurrenceID field not found")
cal.write_text(text.replace(old, new, 1))

caldav = Path("caldav/caldav.go")
text = caldav.read_text()
old = (
    "\tcos := make([]caldav.CalendarObject, len(events))\n"
    "\tfor i, event := range events {\n"
    "\t\tco, err := getCalendarObject(b, calId, calKr, event, bootstrap.CalendarSettings)\n"
    "\t\tif err != nil {\n"
    "\t\t\treturn nil, fmt.Errorf(\"caldav/QueryCalendarObjects: error creating calendar object for event %d: (%w)\", i, err)\n"
    "\t\t}\n\n"
    "\t\tcos[i] = *co\n"
    "\t}\n\n"
    "\treturn cos, nil\n"
)
new = (
    "\tvar cos []caldav.CalendarObject\n"
    "\tfor i, event := range events {\n"
    "\t\tco, err := getCalendarObject(b, calId, calKr, event, bootstrap.CalendarSettings)\n"
    "\t\tif err != nil {\n"
    "\t\t\tlog.Printf(\"caldav/QueryCalendarObjects: skipping event %d (ID: %s) due to error: %v\", i, event.ID, err)\n"
    "\t\t\tcontinue\n"
    "\t\t}\n"
    "\t\tcos = append(cos, *co)\n"
    "\t}\n\n"
    "\treturn cos, nil\n"
)
if old not in text:
    raise SystemExit("caldav.go QueryCalendarObjects block not found")
caldav.write_text(text.replace(old, new, 1))
PY
