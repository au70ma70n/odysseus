#!/bin/sh
# Refresh Proton access tokens after 2FA so /keys/salts does not 401.
set -eu
cd /src/ferroxide

python3 <<'PY'
from pathlib import Path

main = Path("cmd/ferroxide/main.go")
text = main.read_text()
needle = "\t\t\t\ta.Scope = scope\n\t\t\t}\n\t\t}\n\n\t\tvar mailboxPassword string"
insert = (
    "\t\t\t\ta.Scope = scope\n"
    "\t\t\t\trefreshed, err := c.AuthRefresh(a)\n"
    "\t\t\t\tif err != nil {\n"
    "\t\t\t\t\tlog.Fatal(err)\n"
    "\t\t\t\t}\n"
    "\t\t\t\t*a = *refreshed\n"
    "\t\t\t\tc.SetAccessToken(refreshed.UID, refreshed.AccessToken)\n"
    "\t\t\t}\n"
    "\t\t}\n\n"
    "\t\tvar mailboxPassword string"
)
if needle not in text:
    raise SystemExit("main.go auth block not found — ferroxide version changed")
main.write_text(text.replace(needle, insert, 1))

pm = Path("protonmail/protonmail.go")
text = pm.read_text()
needle = "\tkeyRing     openpgp.EntityList\n}\n\nfunc (c *Client) setRequestAuthorization"
insert = (
    "\tkeyRing     openpgp.EntityList\n}\n\n"
    "// SetAccessToken updates in-memory session credentials.\n"
    "func (c *Client) SetAccessToken(uid, accessToken string) {\n"
    "\tc.uid = uid\n"
    "\tc.accessToken = accessToken\n"
    "}\n\n"
    "func (c *Client) setRequestAuthorization"
)
if needle not in text:
    raise SystemExit("protonmail.go client block not found")
pm.write_text(text.replace(needle, insert, 1))

auth = Path("protonmail/auth.go")
text = auth.read_text()
needle = "\tauth.PasswordMode = expiredAuth.PasswordMode\n\treturn auth, nil\n}"
insert = (
    "\tauth.PasswordMode = expiredAuth.PasswordMode\n"
    "\tc.uid = auth.UID\n"
    "\tc.accessToken = auth.AccessToken\n"
    "\treturn auth, nil\n}"
)
if needle not in text:
    raise SystemExit("auth.go AuthRefresh block not found")
auth.write_text(text.replace(needle, insert, 1))
PY
