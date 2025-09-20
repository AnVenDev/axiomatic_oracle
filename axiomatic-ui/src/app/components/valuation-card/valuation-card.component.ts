import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictionResponseV2 } from '../../models/property';

@Component({
  standalone: true,
  selector: 'app-valuation-card',
  imports: [CommonModule],
  templateUrl: './valuation-card.component.html',
  styleUrls: ['./valuation-card.component.scss'],
})
export class ValuationCardComponent {
  @Input() res?: PredictionResponseV2 | null;

  // --- Helpers compat (v2: value_eur/interval_eur/model.*  | legacy: metrics.*, model_meta.*) ---
  private get R(): any {
    return (this.res as any) || {};
  }

  get valueEur(): number | null {
    if (typeof this.R.value_eur === 'number') return this.R.value_eur;
    const k = this.R?.metrics?.valuation_k;
    return typeof k === 'number' ? k * 1000 : null;
  }
  get pointPredEur(): number | null {
    if (typeof this.R.value_eur === 'number') return this.R.value_eur; // v2 = point
    const k = this.R?.metrics?.point_pred_k;
    return typeof k === 'number' ? k * 1000 : null;
  }
  get intervalLoEur(): number | null {
    const iv = this.R.interval_eur;
    if (
      Array.isArray(iv) &&
      iv.length === 2 &&
      iv.every((x: any) => typeof x === 'number')
    )
      return iv[0];
    const k = this.R?.metrics?.confidence_low_k;
    return typeof k === 'number' ? k * 1000 : null;
  }
  get intervalHiEur(): number | null {
    const iv = this.R.interval_eur;
    if (
      Array.isArray(iv) &&
      iv.length === 2 &&
      iv.every((x: any) => typeof x === 'number')
    )
      return iv[1];
    const k = this.R?.metrics?.confidence_high_k;
    return typeof k === 'number' ? k * 1000 : null;
  }
  get marginEur(): number | null {
    const m = this.R?.metrics?.ci_margin_k;
    return typeof m === 'number' ? m * 1000 : null;
  }
  get modelName(): string | null {
    return this.R?.model?.name || this.R?.model_meta?.value_model_name || null;
  }
  get modelVersion(): string | null {
    return (
      this.R?.model?.version || this.R?.model_meta?.value_model_version || null
    );
  }
}
