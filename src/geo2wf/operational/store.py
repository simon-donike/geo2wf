"""SQLite state with atomic per-hour results and immutable live observations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .common import digest, encoded, iso, category


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
          PRAGMA journal_mode=WAL;
          PRAGMA busy_timeout=60000;
          CREATE TABLE IF NOT EXISTS storms(id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, url TEXT NOT NULL,
            retrieved_at TEXT NOT NULL, body BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS status(key TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS samples(storm_id TEXT NOT NULL, time TEXT NOT NULL,
            kind TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL,
            PRIMARY KEY(storm_id,time,kind,model));
          CREATE TABLE IF NOT EXISTS forecasts(storm_id TEXT NOT NULL, time TEXT NOT NULL,
            kind TEXT NOT NULL, model TEXT NOT NULL, body TEXT NOT NULL,
            PRIMARY KEY(storm_id,time,kind,model));
          CREATE TABLE IF NOT EXISTS visuals(storm_id TEXT NOT NULL, time TEXT NOT NULL,
            version TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL,
            PRIMARY KEY(storm_id,time,version));
        """
        )

    def close(self):
        self.db.close()

    @contextmanager
    def read_snapshot(self):
        """A catalog and all of its objects observe the same committed state."""
        self.db.execute("BEGIN")
        try:
            yield
        finally:
            self.db.rollback()

    def put_status(self, key, body):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO status VALUES (?,?)",
                (key, encoded(body).decode()),
            )

    def get_status(self, key):
        row = self.db.execute("SELECT body FROM status WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def snapshot(self, url, body, retrieved_at=None):
        key = digest(body)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO sources VALUES (?,?,?,?)",
                (key, url, iso(retrieved_at), body),
            )
        return key

    def source(self, key):
        row = self.db.execute(
            "SELECT id,url,retrieved_at FROM sources WHERE id=?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def storm(self, sid):
        row = self.db.execute("SELECT body FROM storms WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def storms(self):
        return [
            json.loads(r[0])
            for r in self.db.execute("SELECT body FROM storms ORDER BY id")
        ]

    def put_storm(self, body):
        levels = [
            category(f["wind_ms"])
            for f in body.get("track", [])
            if f.get("wind_ms") is not None
        ]
        if levels:
            body = {**body, "peak_official_category": max(levels)}
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO storms VALUES (?,?)",
                (body["id"], encoded(body).decode()),
            )

    def sample(self, sid, time, kind, model):
        row = self.db.execute(
            "SELECT body FROM samples WHERE storm_id=? AND time=? AND kind=? AND model=?",
            (sid, iso(time), kind, model),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put_sample(self, body, retry=False):
        key = (body["storm_id"], body["time"], body["kind"], body["model_version"])
        previous = self.sample(*key)
        if previous and (previous["status"] == "ready" or not retry):
            return
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?)",
                (*key, body["status"], encoded(body).decode()),
            )

    def samples(self, sid, model=None):
        query = "SELECT body FROM samples WHERE storm_id=?"
        args = [sid]
        if model:
            query += " AND model=?"
            args.append(model)
        return [
            json.loads(r[0])
            for r in self.db.execute(query + " ORDER BY time,kind", args)
        ]

    def put_forecast(self, body):
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?,?)",
                (
                    body["storm_id"],
                    body["anchor_time"],
                    body["kind"],
                    body["model_version"],
                    encoded(body).decode(),
                ),
            )

    def forecasts(self, sid):
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT body FROM forecasts WHERE storm_id=? ORDER BY time,kind", (sid,)
            )
        ]

    def visuals(self, sid, version):
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT body FROM visuals WHERE storm_id=? AND version=? ORDER BY time",
                (sid, version),
            )
        ]

    def put_visual(self, body):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO visuals VALUES (?,?,?,?,?)",
                (
                    body["storm_id"],
                    body["time"],
                    body["version"],
                    body["status"],
                    encoded(body).decode(),
                ),
            )

    def retain(self, cutoff):
        """Keep forecast context separately from the public retention window."""
        with self.db:
            counts = {}
            for table in ("samples", "forecasts", "visuals"):
                counts[table] = self.db.execute(
                    f"DELETE FROM {table} WHERE time < ?", (iso(cutoff),)
                ).rowcount
            for storm in self.storms():
                if not storm.get("active") and storm.get("end", "") < iso(cutoff):
                    self.db.execute("DELETE FROM storms WHERE id=?", (storm["id"],))
                else:
                    track = storm.get("track", [])
                    earlier = [fix for fix in track if fix["time"] < iso(cutoff)]
                    storm["track"] = earlier[-1:] + [
                        fix for fix in track if fix["time"] >= iso(cutoff)
                    ]
                    self.db.execute(
                        "UPDATE storms SET body=? WHERE id=?",
                        (encoded(storm).decode(), storm["id"]),
                    )
            references = {(self.get_status("discovery") or {}).get("snapshot")}
            for storm in self.storms():
                references.update(
                    storm.get(key) for key in ("track_snapshot", "advisory_snapshot")
                )
            for row in self.db.execute("SELECT body FROM samples"):
                record = json.loads(row[0])
                references.add(record.get("source_snapshot"))
                references.update(record.get("source_snapshots", []))
            counts["sources"] = 0
            for row in self.db.execute("SELECT id FROM sources").fetchall():
                if row[0] not in references:
                    counts["sources"] += self.db.execute(
                        "DELETE FROM sources WHERE id=?", (row[0],)
                    ).rowcount
        return counts
