import { Injectable } from '@angular/core';
import {
  HttpClient,
  HttpErrorResponse,
  HttpParams,
} from '@angular/common/http';
import { firstValueFrom, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { PropertyRequest, PredictionResponseV2 } from '../models/property';
import { VerifyResponse } from '../types/ai-oracle.types';

type AnyDict = Record<string, any>;

@Injectable({ providedIn: 'root' })
export class ApiService {
  /**
   * Legge la config del proxy dagli environments.
   * Supporta sia:
   *  - environment['/api'].context  (es. "/api")  → usa path relativo (favorevole all'interceptor)
   *  - environment['/api'].target   (es. "http://127.0.0.1:8000") → assoluto (fallback)
   */
  private readonly proxyCfg: AnyDict = (environment as AnyDict)['/api'] ?? {};
  private readonly baseUrl: string = (
    this.proxyCfg['context'] || // preferisci path relativo (proxy)
    this.proxyCfg['basePath'] || // eventuale alias custom
    '/api'
  ) // fallback sicuro
    .replace(/\/$/, ''); // rimuovi trailing slash

  // In rari casi si vuole ignorare il proxy e colpire direttamente il target:
  private readonly absoluteTarget: string | undefined = this.proxyCfg['target'];

  constructor(private http: HttpClient) {}

  /** Join robusto che garantisce un solo slash */
  private u(path: string, absolute = false): string {
    const base =
      absolute && this.absoluteTarget ? this.absoluteTarget : this.baseUrl;
    const left = base.replace(/\/$/, '');
    const right = path.startsWith('/') ? path : `/${path}`;
    return `${left}${right}`;
  }

  // ---------------- Public API ----------------

  health(): Promise<any> {
    return firstValueFrom(
      this.http.get(this.u('/health')).pipe(catchError(this.handleError))
    );
  }

  listModels(asset = 'property'): Promise<any> {
    return firstValueFrom(
      this.http
        .get(this.u(`/models/${asset}`))
        .pipe(catchError(this.handleError))
    );
  }

  modelHealth(asset = 'property', task = 'value_regressor'): Promise<any> {
    return firstValueFrom(
      this.http
        .get(this.u(`/models/${asset}/${task}/health`))
        .pipe(catchError(this.handleError))
    );
  }

  refreshModel(asset = 'property', task = 'value_regressor'): Promise<any> {
    return firstValueFrom(
      this.http
        .post(this.u(`/models/${asset}/${task}/refresh`), {})
        .pipe(catchError(this.handleError))
    );
  }

  /**
   * Predizione immobile (con opzione publish=true per notarizzazione on-chain).
   * L’interceptor aggiunge l’Authorization header quando necessario.
   */
  // v2: predict con opzioni publish/attestation_only via querystring.
  predictPropertyV2(
    data: PropertyRequest,
    opts?: { publish?: boolean; attestationOnly?: boolean }
  ): Promise<PredictionResponseV2> {
    let params = new HttpParams();
    if (opts?.publish) params = params.set('publish', '1');
    if (opts?.attestationOnly) params = params.set('attestation_only', '1');
    return firstValueFrom(
      this.http
        .post<PredictionResponseV2>(this.u('/predict/property'), data, {
          params,
        })
        .pipe(catchError(this.handleError))
    );
  }

  /**
   * Verifica one-click: controlla che la nota on-chain (txid) corrisponda ai campi della predizione.
   * Puoi passare solo il txid oppure anche il payload di predizione (o subset) per verifica più stretta.
   */
  verifyTx(
    txid: string,
    prediction?: Partial<PredictionResponseV2>,
    expectedSha256?: string
  ): Promise<VerifyResponse> {
    const body: any = { txid };
    if (prediction) body.prediction = prediction;
    if (expectedSha256) body.expected_sha256 = expectedSha256;

    return firstValueFrom(
      this.http
        .post<VerifyResponse>(this.u('/verify'), body)
        .pipe(catchError(this.handleError))
    );
  }

  /** Scarica l’audit bundle ZIP per rid */
  downloadAuditZip(rid: string): Promise<Blob> {
    return firstValueFrom(
      this.http
        .get(this.u(`/audit/${encodeURIComponent(rid)}`), {
          responseType: 'arraybuffer' as const,
        })
        .pipe(
          map((buf) => new Blob([buf], { type: 'application/zip' })), // zip MIME opzionale
          catchError(this.handleError)
        )
    );
  }

  /** Costruisce URL explorer dalla rete (fallback a defaultNetwork) */
  explorerUrl(
    txid: string,
    network?: 'mainnet' | 'testnet' | 'betanet' | 'sandbox' | null
  ): string | null {
    const n =
      (network as any) || (environment as any).defaultNetwork || 'testnet';
    const map = (environment as any).explorers || {};
    const base = map[n];
    return base ? `${base}${txid}` : null;
  }

  private handleError(error: HttpErrorResponse) {
    console.error('[API ERROR]', error);
    return throwError(() => error);
  }

  // ---------------- Logs ----------------

  getApiLogs(): Promise<any[]> {
    return firstValueFrom(
      this.http
        .get<any[]>(this.u('/logs/api'))
        .pipe(catchError(this.handleError))
    );
  }

  getPublishedLogs(): Promise<any[]> {
    return firstValueFrom(
      this.http
        .get<any[]>(this.u('/logs/published'))
        .pipe(catchError(this.handleError))
    );
  }

  getDetailReports(): Promise<any[]> {
    return firstValueFrom(
      this.http
        .get<any[]>(this.u('/logs/detail_reports'))
        .pipe(catchError(this.handleError))
    );
  }
}
