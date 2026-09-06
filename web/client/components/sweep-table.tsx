"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { fetchJSON, media, parseJSONL } from "@/lib/vesper";

// Sweep report: summary from report.json, per-variant table from results.jsonl.
// A failed variant links to its replayable run when the row names one.

const OUTCOME_KEYS = ["success", "ok", "passed", "pass"];
const RUN_KEYS = ["run", "run_id", "runId", "replay"];

type Row = Record<string, unknown>;

function outcome(r: Row): boolean | undefined {
  for (const k of OUTCOME_KEYS) {
    if (typeof r[k] === "boolean") return r[k] as boolean;
  }
  const o = r.outcome ?? r.status ?? r.result;
  if (typeof o === "string") return !/fail|error|crash|timeout/i.test(o);
  if (typeof o === "boolean") return o;
  return undefined;
}

function runLink(r: Row): string | undefined {
  for (const k of RUN_KEYS) {
    if (typeof r[k] === "string" && r[k]) return r[k] as string;
  }
  return undefined;
}

const fmtCell = (v: unknown) =>
  typeof v === "number"
    ? Number.isInteger(v) ? String(v) : v.toFixed(3)
    : typeof v === "object"
      ? JSON.stringify(v)
      : String(v ?? "—");

export function SweepTable({ runId, files }: { runId: string; files: string[] }) {
  const [report, setReport] = useState<Row | null>(null);
  const [rows, setRows] = useState<Row[] | null>(null);

  useEffect(() => {
    let alive = true;
    if (files.includes("report.json"))
      fetchJSON<Row>(media(runId, "report.json")).then((d) => alive && setReport(d));
    if (files.includes("results.jsonl"))
      fetch(media(runId, "results.jsonl"))
        .then((r) => (r.ok ? r.text() : ""))
        .then((t) => alive && setRows(parseJSONL(t)))
        .catch(() => alive && setRows([]));
    return () => {
      alive = false;
    };
  }, [runId, files]);

  const cols = useMemo(() => {
    if (!rows?.length) return [];
    const keys = new Set<string>();
    rows.forEach((r) => Object.keys(r).forEach((k) => keys.add(k)));
    return [...keys];
  }, [rows]);

  const summary = useMemo(() => {
    if (!report) return [];
    return Object.entries(report).filter(
      ([, v]) => typeof v === "number" || typeof v === "string" || typeof v === "boolean",
    );
  }, [report]);

  return (
    <div>
      {summary.length > 0 && (
        <div className="flex flex-wrap gap-x-5 gap-y-1 border-b border-border px-4 py-2.5 text-xs">
          {summary.map(([k, v]) => (
            <span key={k} className="tabular-nums text-secondary-foreground">
              <span className="text-muted-foreground">{k} </span>
              {fmtCell(v)}
            </span>
          ))}
        </div>
      )}
      {!files.includes("results.jsonl") ? (
        <div className="p-4 text-sm text-muted-foreground">no per-variant results recorded</div>
      ) : rows === null ? (
        <div className="p-4 text-sm text-muted-foreground">loading variants…</div>
      ) : rows.length === 0 ? (
        <div className="p-4 text-sm text-muted-foreground">no per-variant results</div>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                {cols.map((c) => (
                  <TableHead key={c} className="whitespace-nowrap text-xs">
                    {c}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r, i) => {
                const ok = outcome(r);
                const link = runLink(r);
                return (
                  <TableRow
                    key={i}
                    className={ok === false ? "bg-[#d03b3b]/10 hover:bg-[#d03b3b]/15" : undefined}
                  >
                    {cols.map((c) => {
                      const isRunCol = RUN_KEYS.includes(c) && typeof r[c] === "string";
                      return (
                        <TableCell
                          key={c}
                          className="whitespace-nowrap text-xs tabular-nums text-secondary-foreground"
                        >
                          {isRunCol && link ? (
                            <Link
                              href={`/runs?run=${encodeURIComponent(link)}`}
                              className="text-[#3987e5] underline-offset-2 hover:underline"
                            >
                              {fmtCell(r[c])} ↗
                            </Link>
                          ) : (
                            fmtCell(r[c])
                          )}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}
