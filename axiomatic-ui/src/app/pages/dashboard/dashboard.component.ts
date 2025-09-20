import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { HttpErrorResponse } from '@angular/common/http';

import { ApiService } from '../../services/api.service';
import { PredictionResponseV2, PropertyRequest } from '../../models/property';

// nuovi componenti UI
import { PropertyFormComponent } from '../../components/property-form/property-form.component';
import { ValuationCardComponent } from '../../components/valuation-card/valuation-card.component';
import { PublishDetailsComponent } from '../../components/publish-details/publish-details.component';
import { QuickVerifyComponent } from '../../components/quick-verify/quick-verify.component';
import {
  WorkflowStep,
  WorkflowStatus,
} from '../../components/workflow-tracker/workflow-tracker.component';
import { MatIconModule } from '@angular/material/icon';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatTooltipModule } from '@angular/material/tooltip';
import { VerifyResponse } from '../../types/ai-oracle.types';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    PropertyFormComponent,
    ValuationCardComponent,
    PublishDetailsComponent,
    QuickVerifyComponent,
    MatIconModule,
    MatSlideToggleModule,
    MatTooltipModule,
  ],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent {
  loading = false;
  publishing = false;

  attOnly = false;

  formValue: PropertyRequest = {} as PropertyRequest;
  lastPayload: PropertyRequest | null = null;

  response: PredictionResponseV2 | null = null;
  errorMessage: string | null = null;

  txId: string | null = null;
  quickVerifyTxid = '';
  explorerBase: string | null = null;

  steps: WorkflowStep[] = [
    { key: 'request', label: 'Send to Oracle', status: 'todo' },
    { key: 'publish', label: 'Publish on-chain', status: 'todo' },
    { key: 'verify', label: 'Verify', status: 'todo' },
    { key: 'download', label: 'Download JSON', status: 'todo' },
  ];

  constructor(private api: ApiService, private snack: MatSnackBar) {}

  private setStep(
    key: 'request' | 'publish' | 'verify' | 'download',
    status: WorkflowStatus,
    meta?: any
  ) {
    this.steps = this.steps.map((s) =>
      s.key === key ? { ...s, status, meta } : s
    );
  }

  // ---------- Helpers 1 ----------
  private statusOf(key: 'request' | 'publish' | 'verify' | 'download') {
    return this.steps.find((s) => s.key === key)?.status;
  }
  statusClass(key: 'request' | 'publish' | 'verify' | 'download') {
    const st = this.statusOf(key);
    return {
      success: st === 'success',
      error: st === 'error',
    };
  }

  get canSend(): boolean {
    return !this.loading && !this.publishing;
  }

  get canPublish(): boolean {
    return !!this.response && !!this.lastPayload && !this.publishing;
  }

  get canVerify(): boolean {
    const st = this.response?.publish?.status;
    const ok = st === 'ok' || st === 'success';
    return ok && !!this.txId && !this.loading && !this.publishing;
  }

  get canDownload(): boolean {
    return !!this.response && !this.publishing;
  }

  scrollToVerify(): void {
    document
      .getElementById('verify-section')
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  stepDone(key: 'request' | 'publish' | 'verify' | 'download') {
    return this.steps.find((s) => s.key === key)?.status === 'success';
  }

  stepError(key: 'request' | 'publish' | 'verify' | 'download') {
    return this.steps.find((s) => s.key === key)?.status === 'error';
  }

  isActive(key: 'request' | 'publish' | 'verify' | 'download') {
    if (key === 'request')
      return !this.response && !this.loading && !this.publishing;
    if (key === 'publish') return !!this.response && !this.response?.publish;
    if (key === 'verify') return this.response?.publish?.status === 'success';
    if (key === 'download') return !!this.response && !this.publishing;
    return false;
  }

  // ---------- Actions ----------

  async sendRequest(): Promise<void> {
    this.errorMessage = null;
    this.response = null;
    this.txId = null;
    this.quickVerifyTxid = '';
    this.setStep('request', 'todo');
    this.setStep('publish', 'todo');
    this.setStep('verify', 'todo');
    this.setStep('download', 'todo');
    this.loading = true;

    try {
      const res = await this.api.predictPropertyV2(this.formValue, {
        publish: false,
        attestationOnly: false,
      });
      this.response = res;
      this.lastPayload = { ...this.formValue };
      this.setStep('request', 'success');
      const net = (res.publish as any)?.network as any;
      this.explorerBase =
        this.api.explorerUrl('DUMMY', net)?.replace(/[^/]+$/, '') || null;
    } catch (err) {
      this.setStep('request', 'error');
      const e = err as HttpErrorResponse;
      const detail =
        (e.error && (e.error.detail || e.error)) ||
        e.message ||
        'Unknown error';
      this.errorMessage = this.translateValidationError(detail);
      this.response = null;
    } finally {
      this.loading = false;
    }
  }

  async publishToBlockchain(): Promise<void> {
    if (!this.lastPayload) {
      this.errorMessage = 'No payload to publish. Send a prediction first.';
      this.snack.open(this.errorMessage, 'OK', {
        duration: 3000,
        horizontalPosition: 'right',
        verticalPosition: 'top',
      });
      return;
    }

    this.publishing = true;
    this.errorMessage = null;

    try {
      const res = await this.api.predictPropertyV2(this.lastPayload, {
        publish: true,
        attestationOnly: false,
      });

      this.response = res;
      this.txId = res.blockchain_txid || res.publish?.txid || null;

      if (res.publish?.status === 'success' || res.publish?.status === 'ok') {
        const txShort = this.txId ? ` (TX ${this.txId.slice(0, 10)}…)` : '';
        this.snack.open('Publish completato su Algorand ✅' + txShort, 'OK', {
          duration: 3500,
          horizontalPosition: 'right',
          verticalPosition: 'top',
        });
        this.setStep('publish', 'success', { txid: this.txId });
        this.quickVerifyTxid = this.txId || '';
        const net = (res.publish as any)?.network as any;
        this.explorerBase =
          this.api.explorerUrl('DUMMY', net)?.replace(/[^/]+$/, '') || null;
      } else {
        this.setStep('publish', 'error');
        const msg = res.publish?.error
          ? `Publish error: ${res.publish.error}`
          : 'Publish failed';
        this.errorMessage = msg;
        this.snack.open(msg, 'OK', {
          duration: 4000,
          horizontalPosition: 'right',
          verticalPosition: 'top',
        });
      }
    } catch (err) {
      this.setStep('publish', 'error');
      const e = err as HttpErrorResponse;

      if (e.status === 429)
        this.errorMessage = 'Too much traffic, please retry shortly.';
      else if (e.status === 413) this.errorMessage = 'Payload too large.';
      else {
        const detail =
          (e.error && (e.error.detail || e.error)) ||
          e.message ||
          'Unknown error';
        this.errorMessage = this.translateValidationError(detail);
      }

      this.snack.open(this.errorMessage || 'Publish error', 'OK', {
        duration: 4000,
        horizontalPosition: 'right',
        verticalPosition: 'top',
      });
    } finally {
      this.publishing = false;
    }
  }

  async onVerifyTx(tx: string): Promise<void> {
    try {
      const expected = this.response?.publish?.note_sha256 || undefined;
      const p1 = (this.response as any)?.attestation?.p1;
      const payload: any = p1
        ? { attestation: { p1 }, asset_id: this.response?.asset_id }
        : this.response || undefined; // fallback

      const res = await this.api.verifyTx(tx, payload, expected);

      if (res?.verified) {
        this.setStep('verify', 'success', {
          txid: tx,
          round: (res as any)?.confirmed_round,
          note: (res as any)?.note_sha256,
        });
        this.snack.open('Verification passed ✅', 'OK', { duration: 2500 });
      } else {
        this.setStep('verify', 'error', { txid: tx, reason: res?.reason });
        this.snack.open(
          `Verification failed: ${res?.reason || 'mismatch'}`,
          'OK',
          {
            duration: 3500,
          }
        );
      }
    } catch (e) {
      console.error('Verify error', e);
      this.setStep('verify', 'error', { txid: tx });
      this.snack.open('Verification error', 'OK', { duration: 3000 });
    }
  }

  // ---------- Helpers 2 ----------

  getExplorerUrl(txId: string): string | null {
    const net = (this.response as any)?.publish?.network as any;
    return this.api.explorerUrl(txId, net);
  }

  openExplorer(): void {
    if (this.txId) {
      const url = this.getExplorerUrl(this.txId);
      if (url) window.open(url, '_blank');
      return;
    }

    const base =
      this.explorerBase ||
      (this.api as any).explorerUrl?.('DUMMY')?.replace(/[^/]$/, '');
    if (base) window.open(base, '_blank');
  }

  async downloadResponse(): Promise<void> {
    const rid = (this.response as any)?.audit_bundle?.rid as string | undefined;
    if (rid) {
      try {
        const blob = await this.api.downloadAuditZip(rid);
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `audit_${rid}.zip`;
        a.click();
        URL.revokeObjectURL(a.href);
        this.setStep('download', 'success', { rid });
        return;
      } catch {}
    }
    // fallback
    const filename = `${this.response!.asset_id || 'response'}.json`;
    const blobJson = new Blob([JSON.stringify(this.response, null, 2)], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blobJson);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
    this.setStep('download', 'success', { json: true });
  }

  // ---------- Validation error translation ----------
  translateValidationError(detail: unknown): string {
    if (Array.isArray(detail)) {
      return this.formatErrorList(
        (detail as any[]).map((e) => ({
          field: e?.loc?.[1] || 'unknown field',
          message: e?.msg || 'invalid value',
        }))
      );
    } else if (typeof detail === 'string') {
      if (
        detail.includes('validation error') &&
        detail.includes('PropertyPredictRequest')
      ) {
        return this.parsePydanticError(detail);
      } else {
        return `Error: ${detail}`;
      }
    }
    return 'Unknown validation error.';
  }

  private parsePydanticError(errorString: string): string {
    const errors: Array<{ field: string; message: string }> = [];
    const valueErrorRegex = /Value error,\s*(.+?)\s*\[type=value_error/g;
    let m: RegExpExecArray | null;
    while ((m = valueErrorRegex.exec(errorString)) !== null) {
      const msg = m[1].trim();
      if (msg.includes('floor must be') && msg.includes('building_floors')) {
        errors.push({
          field: this.translateFieldName('floor'),
          message: 'must be less than the number of building floors',
        });
      } else if (msg.includes('energy_class') && msg.includes('one of')) {
        errors.push({
          field: this.translateFieldName('energy_class'),
          message: 'must be one of: A, B, C, D, E, F, G',
        });
      } else {
        errors.push({ field: 'Validation', message: msg });
      }
    }

    const fieldErrorRegex =
      /^(\w+)\s*\n\s*Input should be (.+?)(?=\s*\[type=)/gm;
    let f: RegExpExecArray | null;
    while ((f = fieldErrorRegex.exec(errorString)) !== null) {
      const field = f[1];
      const raw = f[2].trim();
      const already = errors.some(
        (e) => e.field === this.translateFieldName(field)
      );
      if (!already)
        errors.push({
          field: this.translateFieldName(field),
          message: this.translateMessage(raw),
        });
    }
    return this.formatErrorList(errors);
  }

  private formatErrorList(
    errors: Array<{ field: string; message: string }>
  ): string {
    if (errors.length === 0) return 'Validation error.';
    const header = `Found ${errors.length} validation error${
      errors.length > 1 ? 's' : ''
    }:\n`;
    return (
      header +
      errors.map((e, i) => `${i + 1}. ${e.field}: ${e.message}`).join('\n')
    );
  }

  private translateFieldName(field: string): string {
    const m: Record<string, string> = {
      location: 'Location',
      size_m2: 'Size (m²)',
      rooms: 'Rooms',
      bathrooms: 'Bathrooms',
      year_built: 'Year Built',
      floor: 'Floor',
      building_floors: 'Building Floors',
      has_elevator: 'Elevator',
      has_garden: 'Garden',
      has_balcony: 'Balcony',
      garage: 'Garage',
      energy_class: 'Energy Class',
      humidity_level: 'Humidity Level (%)',
      temperature_avg: 'Average Temperature (°C)',
      noise_level: 'Noise Level (dB)',
      air_quality_index: 'Air Quality Index',
      age_years: 'Age (years)',
    };
    return m[field] || field;
  }

  private translateMessage(message: string): string {
    const map: Record<string, string> = {
      'a valid string': 'must be a valid string',
      'a valid integer': 'must be a valid integer',
      'a valid number': 'must be a valid number',
      'unable to parse string as an integer': 'must be a valid integer',
      'unable to parse string as a number': 'must be a valid number',
      'field required': 'is required',
      'greater than 0': 'must be greater than 0',
      'greater than or equal to 0': 'must be at least 0',
      'greater than or equal to 1': 'must be at least 1',
      'greater than or equal to 1800': 'must be at least 1800',
      'less than or equal to 1': 'must be 0 or 1 (0=No, 1=Yes)',
      'less than or equal to 100': 'cannot exceed 100%',
      'less than or equal to 150': 'cannot exceed 150 dB',
      'less than or equal to 500': 'cannot exceed 500',
    };
    for (const [eng, it] of Object.entries(map)) {
      if (message === eng || message.includes(eng)) return it;
    }
    const yearMatch = message.match(/less than or equal to (\d{4})/);
    if (yearMatch) return `cannot be later than ${yearMatch[1]}`;
    if (
      message.includes('a valid integer') &&
      message.includes('unable to parse')
    )
      return 'must be a valid integer';
    if (
      message.includes('a valid number') &&
      message.includes('unable to parse')
    )
      return 'must be a valid number';
    return message;
  }
}
