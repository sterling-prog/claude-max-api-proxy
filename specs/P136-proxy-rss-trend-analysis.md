# P136 — Proxy RSS Trend Analysis

**Status:** Pending (awaiting 72h of telemetry data; collection restarted 2026-05-20 after PATH fix)

## Goal

Determine if `claude-max-api-proxy` has a real memory leak that warrants a fix.

## Trigger

After 72h of telemetry (≥ 288 data points in `~/.openclaw/logs/proxy-rss-trend.log`).

**Next eligible:** 2026-05-23 ~05:10 UTC (72h from when collection was fixed).

## Telemetry

- **Script:** `~/.openclaw/scripts/log-proxy-rss.sh`
- **Log:** `~/.openclaw/logs/proxy-rss-trend.log`
- **Schedule:** Every 15 minutes via system crontab
- **Format:** `2026-05-07T09:53:00Z rss=50331648` (bytes; ÷1048576 for MB)

## Analysis Steps

1. Parse `~/.openclaw/logs/proxy-rss-trend.log`
2. Summarize RSS over time: min/max/mean, per-hour averages
3. Check for monotonic growth pattern (leak signature)
4. Check for sawtooth pattern (healthy: grows under load, drops on sweep/restart)
5. Compare pre/post nightly 03:30 PM2 restart values to confirm full release on restart
6. If leak confirmed: identify root cause (pool process refs, emitter leaks, etc.)
7. File fix spec if needed; close P136 if no leak found

## Threshold

**Real leak:** RSS grows >20% week-over-week without a restart.

## Fix Applied

The telemetry cron was logging `N/A` for all entries since 2026-05-10 due to a PATH issue.
`pm2` requires `node` (from linuxbrew) but cron runs with a minimal PATH that omits it.
Fix: added explicit `export PATH=".../npm-global/bin:/home/linuxbrew/.linuxbrew/bin:..."` at the
top of `~/.openclaw/scripts/log-proxy-rss.sh`. Verified 2026-05-20; data collection confirmed good.

## Context

- gpu1 host RAM: 62 GB total, ~46 GB available as of 2026-05-06
- Proxy Node process baseline: ~48–87 MB RSS (lean; Claude subprocess RSS is the bulk consumer)
- Each Claude pool process holds 800 MB–1.8 GB RSS (expected; model context loaded in memory)
- 10 pool processes (6 opus + 4 sonnet) = ~8–12 GB committed to subprocesses; well within headroom
- No OOM fix in OpenClaw 2026.5.2–2026.5.5; upgrading won't address memory issues
