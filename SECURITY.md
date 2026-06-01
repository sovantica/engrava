# Security Policy

Full vulnerability-disclosure policy lives on the website:

- **Policy:** <https://engrava.ai/security>
- **Machine-readable (RFC 9116):** <https://engrava.ai/.well-known/security.txt>

This file is the GitHub-side pointer so that the "Report a vulnerability"
surface in the Security tab resolves. The website is the source of truth
and is updated when the policy changes; if the two ever diverge, trust
the website.

## In scope

- The `engrava` Python package on PyPI and its source in this repository.
- Engrava Pro.
- The `engrava.ai` website, including the `/docs` subpath once live.

## Out of scope

- Sovantica-wide infrastructure — email **security@sovantica.ai** once
  the studio policy is live.
- Third-party services that engrava integrates with — report upstream
  to the vendor.
- Rate-limit or denial-of-service findings on the public static site —
  `engrava.ai` has no server-side state to protect.

## Reporting

Preferred channel: email **security@engrava.ai** with:

- A short description of the issue and its impact.
- Reproduction steps, affected version (`pip show engrava`) or commit
  SHA, and a synthetic proof-of-concept if you have one.
- Your preferred contact and whether you want public credit in the
  advisory.

Do not post reproduction steps or proof-of-concept payloads in a public
issue or pull request while the finding is open.

## Out-of-band channel

If email is compromised or unreachable, open a private
[GitHub Security Advisory][advisories]. Do not open a public issue for
a live vulnerability.

[advisories]: https://github.com/sovantica/engrava/security/advisories

## What to expect

- Initial acknowledgment within three business days.
- A status update within ten business days with a triage severity and
  a rough fix timeline.
- Default embargo window: ninety days from initial report, extendable
  by mutual agreement for findings that require a schema migration or
  cross-ecosystem coordination.
- Credit in the release note and, if the finding warrants a CVE, in the
  advisory. Anonymous reports are equally welcome.

engrava is developed by a small team. Complex fixes can take weeks. You
will be told what is happening and when a release is expected, rather
than left on a silent thread.

See <https://engrava.ai/security> for the full policy surface.
