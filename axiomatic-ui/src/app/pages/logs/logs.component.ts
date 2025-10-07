import { AfterViewInit, Component, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTableModule, MatTableDataSource } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTabsModule } from '@angular/material/tabs';
import { MatPaginator, MatPaginatorModule } from '@angular/material/paginator';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatTooltipModule } from '@angular/material/tooltip';

import { ApiService } from '../../services/api.service';
import { PredictionResponseV2 } from '../../models/property';

interface RequestLogMinimal {
  asset_id?: string;
}

interface ApiLogEntry {
  event?: string;
  asset_type?: string;
  request?: RequestLogMinimal;
  response?: PredictionResponseV2 & {
    latency_ms?: number;
    model?: { name?: string; version?: string };
    error?: unknown;
  };
  _logged_at?: string;
  timestamp?: string;
  logged_at?: string;
}

interface PublishedEntry {
  asset_id?: string;
  blockchain_txid?: string;
  asa_id?: string | number;
  logged_at?: string;
  note_hash?: string;
  confirmed_round?: number | string;
  status?: 'success' | 'error' | 'pending' | string;
  finality_s?: number;
  network?: 'mainnet' | 'testnet' | 'betanet' | 'sandbox' | string;
}

interface DetailReport {
  asset_id?: string;
  _logged_at?: string;
  type?: string;
  message?: string;
}

type AnyRow = ApiLogEntry | PublishedEntry | DetailReport;

type RangeKey = '24h' | '7d' | '30d';

@Component({
  selector: 'app-logs',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    MatIconModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatTabsModule,
    MatPaginatorModule,
    MatButtonToggleModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatTooltipModule,
  ],
  templateUrl: './logs.component.html',
  styleUrls: ['./logs.component.scss'],
})
export class LogsComponent implements OnInit, AfterViewInit {
  private rawLogs: ApiLogEntry[] = [];
  private rawPublished: PublishedEntry[] = [];
  private rawReports: DetailReport[] = [];

  logs = new MatTableDataSource<ApiLogEntry>([]);
  published = new MatTableDataSource<PublishedEntry>([]);
  errors = new MatTableDataSource<DetailReport>([]);

  private _logsPaginator?: MatPaginator;
  @ViewChild('logsPaginator')
  set logsPaginatorRef(p: MatPaginator | null) {
    this._logsPaginator = p ?? undefined;
    if (p) this.logs.paginator = p;
  }

  private _publishedPaginator?: MatPaginator;
  @ViewChild('publishedPaginator')
  set publishedPaginatorRef(p: MatPaginator | null) {
    this._publishedPaginator = p ?? undefined;
    if (p) this.published.paginator = p;
  }

  private _errorsPaginator?: MatPaginator;
  @ViewChild('errorsPaginator')
  set errorsPaginatorRef(p: MatPaginator | null) {
    this._errorsPaginator = p ?? undefined;
    if (p) this.errors.paginator = p;
  }

  selectedLog: AnyRow | null = null;

  loading = true;
  error = false;

  logsColumns: string[] = [
    'asset_id',
    'timestamp',
    'model',
    'valuation',
    'latency',
    'actions',
  ];

  range: RangeKey = '24h';
  status: '' | 'success' | 'pending' | 'error' = '';
  model = '';
  models: string[] = [];
  query = '';

  kpi: {
    inferences24h: number;
    pubs24h: number;
    errors24h: number;
    infSuccessRate: number;
    medianFinality: string | number;
    topError: string;
  } | null = null;

  constructor(private api: ApiService) {}

  private asObj(x: any): any {
    if (typeof x === 'string') {
      try {
        return JSON.parse(x);
      } catch {
        /* noop */
      }
    }
    return x ?? undefined;
  }

  private resolveKey(obj: any, key: string): string {
    if (!obj || typeof obj !== 'object') return key;
    const camel = key.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
    const snake = key.replace(/[A-Z]/g, (m) => '_' + m.toLowerCase());
    if (key in obj) return key;
    if (camel in obj) return camel;
    if (snake in obj) return snake;
    return key;
  }

  private fromResp<T = any>(row: any, path: string): T | undefined {
    const get = (root: any, p: string) => {
      let cur = this.asObj(root);
      console.log(cur);
      for (const raw of p.split('.')) {
        if (cur == null) return undefined;
        const k = this.resolveKey(cur, raw);
        cur = cur[k];
      }
      return cur;
    };

    const resp = this.asObj(row?.response);
    const meta = this.asObj(row?.response_meta);
    const flat = this.asObj(row);

    let v = get(resp, path);
    if (v !== undefined) return v as T;

    v = get(meta, path);
    if (v !== undefined) return v as T;

    if (path === 'model_meta.value_model_name') {
      return (get(resp, 'model.name') ?? get(meta, 'model.name')) as T;
    }
    if (path === 'model_meta.value_model_version') {
      return (get(resp, 'model.version') ?? get(meta, 'model.version')) as T;
    }

    return get(flat, path) as T | undefined;
  }

  rowDate(row: any): string | null {
    const ts =
      this.fromResp<string>(row, 'timestamp') ??
      (row && (row._logged_at || row.timestamp)) ??
      null;
    return ts;
  }

  modelName(row: any): string | null {
    return this.fromResp<string>(row, 'model_meta.value_model_name') || null;
  }
  modelVersion(row: any): string | null {
    return this.fromResp<string>(row, 'model_meta.value_model_version') || null;
  }
  valuationK(row: any): number | null {
    const v = this.fromResp<number>(row, 'metrics.valuation_k');
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  }
  uncertaintyK(row: any): number | null {
    const s = this.fromResp<number>(row, 'metrics.uncertainty_k');
    return typeof s === 'number' && Number.isFinite(s) ? s : null;
  }
  latencyMs(row: any): number | null {
    const l = this.fromResp<number>(row, 'metrics.latency_ms');
    return typeof l === 'number' && Number.isFinite(l) ? l : null;
  }

  isSelApi(): boolean {
    return this.isApiLog(this.selectedLog);
  }
  isSelPub(): boolean {
    return this.isPublished(this.selectedLog);
  }
  isSelErr(): boolean {
    return this.isReport(this.selectedLog);
  }

  ovModelName(): string | null {
    if (!this.isSelApi()) return null;
    return this.modelName(this.selectedLog as any) || null;
  }
  ovModelVersion(): string | null {
    if (!this.isSelApi()) return null;
    return this.modelVersion(this.selectedLog as any) || null;
  }
  ovValuationK(): number | null {
    if (!this.isSelApi()) return null;
    return this.valuationK(this.selectedLog as any);
  }
  ovUncertaintyK(): number | null {
    if (!this.isSelApi()) return null;
    return this.uncertaintyK(this.selectedLog as any);
  }
  ovLatencyMs(): number | null {
    if (!this.isSelApi()) return null;
    return this.latencyMs(this.selectedLog as any);
  }
  ovPublishStatus(): string | null {
    if (!this.isSelApi()) return null;
    const row: any = this.selectedLog;
    return (
      row?.response_meta?.publish_status ||
      row?.response?.publish?.status ||
      null
    );
  }
  ovP1Sha(): string | null {
    if (!this.selectedLog) return null;
    const row: any = this.selectedLog;
    if (this.isApiLog(row)) {
      return row?.response?.attestation?.p1_sha256 || null;
    }
    if (this.isPublished(row)) {
      return (row as any).note_hash || (row as any).note_sha256 || null;
    }
    return null;
  }

  selectedStatus(): string | null {
    if (!this.selectedLog) return null;
    if (this.isPublished(this.selectedLog)) {
      return this.selectedLog.status || null;
    }
    if (this.isApiLog(this.selectedLog)) {
      return this.statusOfInference(this.selectedLog as any);
    }
    if (this.isReport(this.selectedLog)) {
      return (this.selectedLog as any).type || 'error';
    }
    return null;
  }

  isApiLog(row: AnyRow | null | undefined): row is ApiLogEntry {
    return (
      !!row &&
      ('response' in row || 'request' in row) &&
      !('blockchain_txid' in row)
    );
  }
  isPublished(row: AnyRow | null | undefined): row is PublishedEntry {
    return (
      !!row &&
      ('blockchain_txid' in row || 'asa_id' in row || 'note_hash' in row)
    );
  }
  isReport(row: AnyRow | null | undefined): row is DetailReport {
    return !!row && ('type' in row || 'message' in row);
  }

  selectedTimestamp(): string {
    if (!this.selectedLog) return '—';
    const ms = this.getTime(this.selectedLog);
    return ms ? new Date(ms).toLocaleString() : '—';
  }
  selectedAssetId(): string {
    return this.selectedLog ? this.getAssetId(this.selectedLog) : '—';
  }
  selectedTxid(): string | null {
    return this.isPublished(this.selectedLog)
      ? this.selectedLog.blockchain_txid || null
      : null;
  }
  selectedNote(): string | null {
    return this.isPublished(this.selectedLog)
      ? this.selectedLog.note_hash || null
      : null;
  }
  selectedRound(): number | string | null {
    return this.isPublished(this.selectedLog)
      ? this.selectedLog.confirmed_round ?? null
      : null;
  }
  selectedRequest(): unknown {
    return this.isApiLog(this.selectedLog)
      ? this.selectedLog.request ?? null
      : null;
  }
  selectedResponse(): unknown {
    return this.isApiLog(this.selectedLog)
      ? this.selectedLog.response ?? null
      : null;
  }

  async ngOnInit(): Promise<void> {
    this.loading = true;
    this.error = false;

    try {
      const [logsData, publishedData, reportsData] = await Promise.all([
        this.api.getApiLogs() as Promise<ApiLogEntry[]>,
        this.api.getPublishedLogs() as Promise<PublishedEntry[]>,
        this.api.getDetailReports() as Promise<DetailReport[]>,
      ]);

      this.rawLogs = [...logsData].sort(
        (a, b) => this.getTime(b) - this.getTime(a)
      );
      this.rawPublished = [...publishedData].sort(
        (a, b) => this.getTime(b) - this.getTime(a)
      );
      this.rawReports = [...reportsData].sort(
        (a, b) => this.getTime(b) - this.getTime(a)
      );

      this.models = Array.from(
        new Set(
          this.rawLogs
            .map((r) => this.fromResp<string>(r, 'model_meta.value_model_name'))
            .filter(Boolean) as string[]
        )
      ).sort();

      this.applyFilters();
      this.loading = false;
    } catch (err) {
      console.error('Error while loading logs:', err);
      this.error = true;
      this.loading = false;
    }
  }

  ngAfterViewInit(): void {
    this.logs.paginator = this._logsPaginator ?? null;
    this.published.paginator = this._publishedPaginator ?? null;
    this.errors.paginator = this._errorsPaginator ?? null;
  }

  private statusOfInference(r: any): 'success' | 'error' | 'pending' {
    if (r?.response?.metrics || r?.response_meta?.metrics) return 'success';
    if (r?.response?.error || r?.event === 'error') return 'error';
    return 'pending';
  }

  private getTime(r: AnyRow): number {
    const ts =
      (r as any)._logged_at ||
      (r as any).timestamp ||
      (r as any).response?.timestamp ||
      (r as any).response_meta?.timestamp ||
      (r as PublishedEntry).logged_at ||
      '';
    const t = ts ? Date.parse(ts as string) : NaN;
    return Number.isNaN(t) ? 0 : t;
  }

  onRangeChange(v: RangeKey) {
    this.range = v;
    this.applyFilters(true);
  }
  onStatusChange(v: '' | 'success' | 'pending' | 'error') {
    this.status = v;
    this.applyFilters(true);
  }
  onModelChange(v: string) {
    this.model = v;
    this.applyFilters(true);
  }
  onSearch(v: string) {
    this.query = (v || '').trim();
    this.applyFilters(true);
  }

  private applyFilters(resetPage = false): void {
    const now = Date.now();
    const windowStart = this.windowStartMs(now, this.range);

    const matchTime = (r: AnyRow) => this.getTime(r) >= windowStart;
    const matchQuery = (text: string | undefined) =>
      !this.query ||
      (text || '').toLowerCase().includes(this.query.toLowerCase());

    let inf = this.rawLogs.filter(matchTime);
    if (this.status)
      inf = inf.filter((r) => this.statusOfInference(r) === this.status);
    if (this.model) {
      inf = inf.filter(
        (r: any) =>
          (r?.response?.model_meta?.value_model_name ||
            r?.response_meta?.model_meta?.value_model_name ||
            '') === this.model
      );
    }
    if (this.query) {
      inf = inf.filter((r: any) =>
        matchQuery(
          r?.response?.asset_id ||
            r?.response_meta?.asset_id ||
            r?.request?.asset_id
        )
      );
    }

    let pubs = this.rawPublished.filter(matchTime);
    if (this.status)
      pubs = pubs.filter((r) => this.statusOfPublication(r) === this.status);
    if (this.query) {
      pubs = pubs.filter(
        (r) =>
          matchQuery(r.asset_id) ||
          matchQuery(r.blockchain_txid) ||
          matchQuery(r.note_hash)
      );
    }

    let errs = this.rawReports.filter(matchTime);
    if (!errs.length) {
      errs = [
        ...this.rawLogs
          .filter((r) => matchTime(r) && this.statusOfInference(r) === 'error')
          .map((r: any) => ({
            asset_id:
              r?.response?.asset_id ||
              r?.response_meta?.asset_id ||
              r?.request?.asset_id,
            _logged_at: r?._logged_at || r?.timestamp,
            type: 'inference',
            message: r?.response?.error
              ? String(r.response.error)
              : 'inference error',
          })),
        ...this.rawPublished
          .filter(
            (r) => matchTime(r) && this.statusOfPublication(r) === 'error'
          )
          .map((r) => ({
            asset_id: r.asset_id,
            _logged_at: r.logged_at,
            type: 'publish',
            message: 'publish error',
          })),
      ];
    }
    if (this.query)
      errs = errs.filter(
        (r) => matchQuery(r.asset_id) || matchQuery(r.message)
      );

    this.logs.data = inf;
    this.published.data = pubs;
    this.errors.data = errs;

    if (resetPage) {
      this._logsPaginator?.firstPage();
      this._publishedPaginator?.firstPage();
      this._errorsPaginator?.firstPage();
    }

    this.kpi = this.computeKpi(windowStart);
  }

  private computeKpi(windowStart: number) {
    const inf = this.rawLogs.filter((r) => this.getTime(r) >= windowStart);
    const pubs = this.rawPublished.filter(
      (r) => this.getTime(r) >= windowStart
    );

    const infTotal = inf.length;
    const infSucc = inf.filter(
      (r) => this.statusOfInference(r) === 'success'
    ).length;
    const pubErrors = pubs.filter(
      (r) => this.statusOfPublication(r) === 'error'
    ).length;

    const finals = pubs
      .map((p) => Number((p as any).finality_s))
      .filter((v) => Number.isFinite(v)) as number[];
    const medianFinality = finals.length ? this.median(finals).toFixed(1) : '—';

    const errMessages: Record<string, number> = {};
    for (const e of this.errors.data) {
      const m = (e.message || 'error').toString();
      errMessages[m] = (errMessages[m] || 0) + 1;
    }
    const topError =
      Object.entries(errMessages).sort((a, b) => b[1] - a[1])[0]?.[0] || '—';

    return {
      inferences24h: infTotal,
      pubs24h: pubs.length,
      errors24h: pubErrors,
      infSuccessRate: infTotal ? Math.round((infSucc / infTotal) * 100) : 0,
      medianFinality,
      topError,
    };
  }

  private median(arr: number[]): number {
    const a = [...arr].sort((x, y) => x - y);
    const mid = Math.floor(a.length / 2);
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }

  private windowStartMs(now: number, range: RangeKey): number {
    const day = 24 * 60 * 60 * 1000;
    switch (range) {
      case '7d':
        return now - 7 * day;
      case '30d':
        return now - 30 * day;
      default:
        return now - 1 * day;
    }
  }

  private statusOfPublication(
    r: PublishedEntry
  ): 'success' | 'error' | 'pending' {
    if (r.status === 'success' || (!!r.blockchain_txid && r.status !== 'error'))
      return 'success';
    if (r.status === 'error') return 'error';
    return 'pending';
  }

  getAssetId(r: AnyRow): string {
    const fromPublished = (r as PublishedEntry).asset_id;
    const fromResp = (r as ApiLogEntry).response?.asset_id;
    const fromRespMeta = (r as any)?.response_meta?.asset_id;
    const fromReq = (r as ApiLogEntry).request?.asset_id;
    return fromPublished || fromResp || fromRespMeta || fromReq || 'unknown';
  }

  selectLog(log: AnyRow): void {
    this.selectedLog = log;
  }

  downloadJSON(row: AnyRow): void {
    const blob = new Blob([JSON.stringify(row, null, 2)], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    const name =
      (row as PublishedEntry).asset_id ||
      (row as PublishedEntry).blockchain_txid ||
      (row as any)?.response_meta?.asset_id ||
      (row as ApiLogEntry).response?.asset_id ||
      (row as ApiLogEntry).request?.asset_id ||
      'log';
    a.href = url;
    a.download = `${name}.json`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  verify(row: AnyRow): void {
    const txid = (row as PublishedEntry).blockchain_txid;
    if (txid) {
      const url = this.getExplorerUrl(
        txid,
        (row as PublishedEntry).network as any
      );
      window.open(url, '_blank', 'noopener');
      return;
    }
    if (
      this.selectedLog &&
      (this.selectedLog as PublishedEntry).blockchain_txid
    ) {
      const t = (this.selectedLog as PublishedEntry).blockchain_txid!;
      const url = this.getExplorerUrl(
        t,
        (this.selectedLog as PublishedEntry).network as any
      );
      window.open(url, '_blank', 'noopener');
    }
  }

  replay(row: AnyRow): void {
    try {
      localStorage.setItem('axiomatic:lastLog', JSON.stringify(row));
      window.location.href = '/dashboard';
    } catch {}
  }

  getValuationK(log: ApiLogEntry): number | null {
    return log.response?.metrics?.valuation_k ?? null;
  }

  getExplorerUrl(txid: string, network?: any): string {
    const fn = (this.api as any).explorerUrl as
      | ((txid: string, net?: any) => string | null | undefined)
      | undefined;
    return fn?.(txid, network) || `https://testnet.peraexplorer.com/tx/${txid}`;
  }
}
