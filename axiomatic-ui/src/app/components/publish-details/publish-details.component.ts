import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictionResponseV2 } from '../../models/property';

@Component({
  standalone: true,
  selector: 'app-publish-details',
  imports: [CommonModule],
  templateUrl: './publish-details.component.html',
  styleUrls: ['./publish-details.component.scss'],
})
export class PublishDetailsComponent {
  @Input() res?: PredictionResponseV2 | null;

  // helpers stato
  isOk(s?: string | null) {
    return s === 'success' || s === 'ok';
  }
  isErr(s?: string | null) {
    return s === 'error';
  }
  isSkipped(s?: string | null) {
    return s === 'skipped' || s === 'not_attempted';
  }

  // on-chain?
  hasOnChain(): boolean {
    const r = this.res;
    const p = r?.publish;
    return !!(
      r?.blockchain_txid ||
      p?.txid ||
      r?.asa_id ||
      p?.asa_id ||
      r?.note_sha256 ||
      p?.note_sha256 ||
      r?.confirmed_round ||
      p?.confirmed_round
    );
  }

  get publish() {
    return this.res?.publish;
  }
  get txid(): string | null {
    return this.res?.blockchain_txid || this.res?.publish?.txid || null;
  }
  get asa(): string | number | null {
    return (
      (this.res?.asa_id as any) ?? (this.res?.publish?.asa_id as any) ?? null
    );
  }

  get noteSha(): string | null {
    return this.res?.note_sha256 || this.publish?.note_sha256 || null;
  }
  get noteSize(): number | null {
    return (
      (this.res?.note_size as any) ?? (this.publish?.note_size as any) ?? null
    );
  }
  get isCompactedDefined(): boolean {
    return (
      typeof ((this.res as any)?.is_compacted ?? this.publish?.is_compacted) ===
      'boolean'
    );
  }
  get isCompacted(): boolean {
    return !!(((this.res as any)?.is_compacted ??
      this.publish?.is_compacted) as boolean);
  }
  get confirmedRound(): number | null {
    return (
      (this.res?.confirmed_round as any) ??
      (this.publish?.confirmed_round as any) ??
      null
    );
  }
  get network(): string | null {
    return (this.publish?.network as any) ?? null;
  }
  get explorerUrl(): string | null {
    return (this.publish?.explorer_url as any) ?? null;
  }
}
