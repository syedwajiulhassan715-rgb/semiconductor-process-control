/*
 * Client-side replay engine for the SECOM monitor: a line-for-line port of
 * secom/monitor.py, so the page can run with no server (the static demo).
 *
 * It mirrors ReplaySession.snapshot() field for field, which lets the page use
 * the same rendering code against either the Python JSON API or this engine.
 * The CUSUM weights and g-chart limits are taken from the exported design
 * rather than recomputed, so every CUSUM value is the same double the Python
 * engine produces; tests/test_monitor_static.py checks that with node.
 *
 * Loadable as a browser script (window.SecomEngine) or a CommonJS module.
 */
(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SecomEngine = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const RECENT_WINDOW = 50;
  const CHANCE_EXCEEDANCE = 0.0027;

  // Python's f"{x:.2f}" for the alarm messages.
  const f2 = (x) => x.toFixed(2);

  class Cusum {
    constructor(h, w) {
      this.h = h;
      this.wFail = w.fail;
      this.wPass = w.pass;
      this.value = 0;
    }
    update(isFail) {
      const inc = isFail ? this.wFail : this.wPass;
      const acc = Math.max(0, this.value + inc);
      const signal = acc > this.h;
      this.value = signal ? 0 : acc;
      return { value: acc, signal };
    }
  }

  class GChart {
    constructor(g) {
      this.ucl = g.ucl;
      this.gap = 0;
      this.seenFail = false;
      this.signalled = false;
    }
    update(isFail) {
      if (isFail) {
        const closed = this.seenFail ? this.gap : null;
        this.seenFail = true;
        this.gap = 0;
        this.signalled = false;
        return { gap: 0, closed_gap: closed, signal: false };
      }
      this.gap += 1;
      const signal = this.seenFail && !this.signalled && this.gap > this.ucl;
      if (signal) this.signalled = true;
      return { gap: this.gap, closed_gap: null, signal };
    }
  }

  class Window {
    constructor(n) { this.n = n; this.items = []; this.sum = 0; }
    push(b) {
      this.items.push(b);
      this.sum += b ? 1 : 0;
      if (this.items.length > this.n) this.sum -= this.items.shift() ? 1 : 0;
    }
    get length() { return this.items.length; }
  }

  class Watch {
    constructor(d) {
      this.lcl = d.watch_lcl;
      this.ucl = d.watch_ucl;
      this.median = d.watch_median;
      this.mad = d.watch_mad;
      this.recent = new Window(RECENT_WINDOW);
    }
    update(raw) {
      const v = raw == null ? NaN : raw;
      const beyond = v < this.lcl || v > this.ucl;
      this.recent.push(beyond);
      const z = this.mad ? (v - this.median) / this.mad : NaN;
      return { value: v, robust_z: z, beyond };
    }
  }

  class AlarmError extends Error {}

  class AlarmBook {
    constructor() { this.alarms = []; this.log = []; }
    openFor(source) {
      for (let i = this.alarms.length - 1; i >= 0; i--) {
        const a = this.alarms[i];
        if (a.source === source && a.acknowledged_run === null) return a;
      }
      return null;
    }
    signal(source, kind, message, run, at) {
      const cur = this.openFor(source);
      if (cur) {
        cur.signals += 1;
        cur.last_signal_run = run;
        this.log.push({ run, at, event: "repeat", alarm_id: cur.id, source,
          message: `repeat signal (${cur.signals} so far)` });
        return cur;
      }
      const a = {
        id: this.alarms.length + 1, source, kind, message, raised_run: run, raised_at: at,
        signals: 1, last_signal_run: run, acknowledged_run: null, acknowledged_at: null, note: "",
      };
      this.alarms.push(a);
      this.log.push({ run, at, event: "raised", alarm_id: a.id, source, message });
      return a;
    }
    acknowledge(id, run, at, note) {
      const a = this.alarms.find((x) => x.id === id);
      if (!a) throw new AlarmError(`no alarm with id ${id}`);
      if (a.acknowledged_run !== null) {
        throw new AlarmError(`alarm ${id} was already acknowledged at run ${a.acknowledged_run}`);
      }
      a.acknowledged_run = run;
      a.acknowledged_at = at;
      a.note = note || "";
      this.log.push({ run, at, event: "acknowledged", alarm_id: id, source: a.source,
        message: note || "acknowledged" });
      return a;
    }
  }

  const asDict = (a) => ({ ...a, status: a.acknowledged_run === null ? "open" : "acknowledged" });
  const finite = (x) => (typeof x === "number" && !Number.isFinite(x) ? null : x);

  /** Expand the exporter's compact columns into one array per field. */
  function decodeRuns(data) {
    const r = data.runs;
    return {
      run_id: r.run_id,
      timestamp: r.timestamp,
      year_week: r.year_week_idx.map((i) => r.year_week_values[i]),
      phase: r.phase_idx.map((i) => r.phase_values[i]),
      is_fail: Array.from(r.is_fail, (c) => c === "1" || c === 1 || c === true),
      watch_value: r.watch_value,
    };
  }

  class Session {
    constructor(design, runs) {
      this.design = design;
      this.runs = runs;
      this.n = runs.run_id.length;
      this.acks = [];
      this.resetMonitors();
    }
    resetMonitors() {
      const d = this.design;
      this.cusum = new Cusum(d.h, d.weights.upper);
      this.cusumLower = new Cusum(d.lower_h, d.weights.lower);
      this.gchart = new GChart(d.gchart);
      this.watch = new Watch(d);
      this.alarms = new AlarmBook();
      this.history = [];
      this.fails = 0;
      this.recent = new Window(RECENT_WINDOW);
    }
    get cursor() { return this.history.length; }
    get finished() { return this.cursor >= this.n; }

    processNext() {
      const i = this.cursor, r = this.runs, d = this.design;
      const fail = r.is_fail[i];
      const run = i + 1, at = String(r.timestamp[i]);
      const c = this.cusum.update(fail);
      const lo = this.cusumLower.update(fail);
      const g = this.gchart.update(fail);
      const w = this.watch.update(r.watch_value[i]);
      this.fails += fail ? 1 : 0;
      this.recent.push(fail);
      if (c.signal) {
        this.alarms.signal("cusum", "degradation",
          `CUSUM ${f2(c.value)} > h = ${f2(d.h)}: failure rate above baseline`, run, at);
      }
      if (lo.signal) {
        this.alarms.signal("cusum_lower", "improvement",
          `lower CUSUM ${f2(lo.value)} > h = ${f2(d.lower_h)}: failure rate below baseline`, run, at);
      }
      const point = {
        run, run_id: r.run_id[i], timestamp: at, year_week: r.year_week[i], phase: r.phase[i],
        is_fail: fail,
        cusum: c.value, cusum_signal: c.signal,
        cusum_lower: lo.value, cusum_lower_signal: lo.signal,
        gap: g.gap, closed_gap: g.closed_gap, gchart_signal: g.signal,
        watch_value: finite(w.value), watch_z: finite(w.robust_z), watch_beyond: w.beyond,
        fail_rate: this.fails / run,
        recent_fail_rate: this.recent.sum / this.recent.length,
      };
      this.history.push(point);
      for (const a of this.acks) {
        if (a.after_run === run) this.alarms.acknowledge(a.alarm_id, run, at, a.note);
      }
      return point;
    }
    step(n) {
      if (n < 0) throw new RangeError("n must be non-negative");
      const out = [];
      for (let k = 0; k < n && !this.finished; k++) out.push(this.processNext());
      return out;
    }
    seek(run) {
      const target = Math.min(Math.max(Math.trunc(run), 0), this.n);
      this.acks = this.acks.filter((a) => a.after_run <= target);
      this.resetMonitors();
      this.step(target);
    }
    reset() {
      this.acks = [];
      this.resetMonitors();
    }
    acknowledge(id, note) {
      if (this.cursor === 0) throw new AlarmError("nothing has run yet");
      const at = String(this.runs.timestamp[this.cursor - 1]);
      const a = this.alarms.acknowledge(id, this.cursor, at, note || "");
      this.acks.push({ alarm_id: id, after_run: this.cursor, note: note || "" });
      return a;
    }
    snapshot(since) {
      since = Math.min(Math.max(Math.trunc(since || 0), 0), this.cursor);
      const last = this.history.length ? this.history[this.history.length - 1] : null;
      const rw = this.watch.recent;
      return {
        cursor: this.cursor,
        n_runs: this.n,
        finished: this.finished,
        current: last,
        watch: {
          sensor: this.design.watch_sensor,
          recent_beyond: rw.sum,
          recent_n: rw.length,
          window: RECENT_WINDOW,
          expected_by_chance: CHANCE_EXCEEDANCE * rw.length,
        },
        open_alarms: this.alarms.alarms.filter((a) => a.acknowledged_run === null).map(asDict),
        alarms: this.alarms.alarms.map(asDict),
        log: this.alarms.log.map((e) => ({ ...e })),
        since,
        history: this.history.slice(since),
      };
    }
  }

  /**
   * The JSON API of monitor_server.py, answered in-process. ``request(path,
   * body)`` returns what the server would, or throws an Error whose message is
   * the server's error text.
   */
  function localApi(data, defaultSpeed) {
    const design = { ...data.design };
    if (defaultSpeed != null) design.default_speed = defaultSpeed;
    const session = new Session(design, decodeRuns(data));
    const int = (body, key, def) => {
      const v = body[key] === undefined ? def : body[key];
      if (v === undefined || v === null) throw new Error(`missing '${key}'`);
      const n = Number(v);
      if (!Number.isInteger(n)) throw new Error(`'${key}' must be an integer`);
      return n;
    };
    const routes = {
      "api/step": (b) => {
        const n = int(b, "n", 1);
        if (n < 0 || n > 2000) throw new Error("'n' must be in [0, 2000]");
        session.step(n);
      },
      "api/seek": (b) => session.seek(int(b, "run")),
      "api/reset": () => session.reset(),
      "api/ack": (b) => {
        const note = b.note === undefined ? "" : b.note;
        if (typeof note !== "string" || note.length > 500) {
          throw new Error("'note' must be a string of at most 500 chars");
        }
        session.acknowledge(int(b, "alarm_id"), note.trim());
      },
    };
    return {
      session,
      request(path, body) {
        const [route, query] = path.replace(/^\//, "").split("?");
        if (body === undefined) {
          if (route === "api/design") return design;
          if (route === "api/state") {
            const since = new URLSearchParams(query || "").get("since") || "0";
            return session.snapshot(int({ since }, "since"));
          }
          throw new Error(`no route /${route}`);
        }
        const action = routes[route];
        if (!action) throw new Error(`no route /${route}`);
        action(body);
        return session.snapshot(int(body, "since", 0));
      },
    };
  }

  return { Session, AlarmError, decodeRuns, localApi, RECENT_WINDOW };
});
