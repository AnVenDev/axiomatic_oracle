import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { ApiService } from '../../services/api.service';
import { PredictionResponseV2, PropertyRequest } from '../../models/property';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatTabsModule } from '@angular/material/tabs';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatInputModule,
    MatTableModule,
    MatPaginatorModule,
    MatTabsModule,
    MatProgressSpinnerModule,
    MatIconModule,
    MatCardModule,
  ],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent {
  jsonInput = '';
  response: PredictionResponseV2 | null = null;
  errorMessage: string | null = null;
  selectedLog: any = null;
  loading = false;
  publishing = false;
  txId: string | null = null;

  // ultimo payload valido inviato (utile per republish)
  private lastPayload: PropertyRequest | null = null;

  constructor(private api: ApiService) {}

  /** Parse JSON dalla textarea con messaggio d’errore chiaro. */
  private parseInput(): PropertyRequest | null {
    try {
      const parsed = JSON.parse(this.jsonInput);
      return parsed as PropertyRequest;
    } catch {
      this.errorMessage = 'Invalid JSON. Please check the syntax.';
      return null;
    }
  }

  /** Invia la richiesta di predizione (senza publish). */
  async sendRequest(): Promise<void> {
    this.errorMessage = null;
    this.response = null;
    this.loading = true;
    this.txId = null;

    const payload = this.parseInput();
    if (!payload) {
      this.loading = false;
      return;
    }

    try {
      const res: PredictionResponseV2 = await this.api.predictProperty(
        payload,
        false
      );
      this.response = res;
      this.lastPayload = payload;
    } catch (err) {
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

  /** Publish on-chain: rilancia la predict con publish=true riusando l’ultimo payload. */
  async publishToBlockchain(): Promise<void> {
    if (!this.lastPayload) {
      this.errorMessage = 'No payload to publish. Send a prediction first.';
      return;
    }
    this.publishing = true;
    this.errorMessage = null;

    try {
      const res: PredictionResponseV2 = await this.api.predictProperty(
        this.lastPayload,
        true
      );
      this.response = res;
      this.txId = res.blockchain_txid || res.publish?.txid || null;
    } catch (err) {
      const e = err as HttpErrorResponse;
      const detail =
        (e.error && (e.error.detail || e.error)) ||
        e.message ||
        'Unknown error';
      this.errorMessage = this.translateValidationError(detail);
    } finally {
      this.publishing = false;
    }
  }

  /** Carica un file JSON e lo posiziona nella textarea (pretty-printed). */
  loadFile(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;

    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(reader.result as string);
        this.jsonInput = JSON.stringify(parsed, null, 2);
        this.errorMessage = null;
      } catch {
        this.errorMessage = 'Invalid JSON file. Please check syntax.';
      }
    };
    reader.readAsText(file);
  }

  /** Scarica l’ultima risposta come JSON prettificato. */
  downloadResponse(): void {
    if (!this.response) return;
    const filename = `${this.response.asset_id || 'response'}.json`;
    const blob = new Blob([JSON.stringify(this.response, null, 2)], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  // ----------------------------------------------------------------------------
  // ERROR TRANSLATIONS (immutati, solo tipizzati)
  // ----------------------------------------------------------------------------

  translateValidationError(detail: unknown): string {
    if (Array.isArray(detail)) {
      // FastAPI/Pydantic array of objects
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

    // model_validator errors
    const valueErrorRegex = /Value error,\s*(.+?)\s*\[type=value_error/g;
    let valueMatch: RegExpExecArray | null;
    while ((valueMatch = valueErrorRegex.exec(errorString)) !== null) {
      const errorMessage = valueMatch[1].trim();

      if (
        errorMessage.includes('floor must be') &&
        errorMessage.includes('building_floors')
      ) {
        errors.push({
          field: this.translateFieldName('floor'),
          message: 'must be less than the number of building floors',
        });
      } else if (
        errorMessage.includes('energy_class must be one of') ||
        (errorMessage.includes('energy_class') &&
          errorMessage.includes('one of'))
      ) {
        errors.push({
          field: this.translateFieldName('energy_class'),
          message: 'must be one of: A, B, C, D, E, F, G',
        });
      } else {
        errors.push({ field: 'Validation', message: errorMessage });
      }
    }

    // standard field errors
    const fieldErrorRegex =
      /^(\w+)\s*\n\s*Input should be (.+?)(?=\s*\[type=)/gm;
    let fieldMatch: RegExpExecArray | null;
    while ((fieldMatch = fieldErrorRegex.exec(errorString)) !== null) {
      const field = fieldMatch[1];
      const rawMessage = fieldMatch[2].trim();
      const alreadyAdded = errors.some(
        (e) => e.field === this.translateFieldName(field)
      );
      if (!alreadyAdded) {
        const message = this.translateMessage(rawMessage);
        errors.push({ field: this.translateFieldName(field), message });
      }
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
    const translations: Record<string, string> = {
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
    return translations[field] || field;
  }

  private translateMessage(message: string): string {
    const translations: Record<string, string> = {
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

    for (const [eng, translated] of Object.entries(translations)) {
      if (message === eng || message.includes(eng)) return translated;
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

  // ----------------------------------------------------------------------------
  // Helpers UI
  // ----------------------------------------------------------------------------

  getExplorerUrl(txId: string): string {
    // TestNet explorer (puoi sostituire con mainnet/dappflow)
    return `https://testnet.peraexplorer.com/tx/${txId}`;
  }

  /**
   * Compat con il template: interpreta `mae` come `ci_margin_k` (margine del CI).
   * Restituisce l'intervallo in EURO (non k€).
   */
  getConfidenceInterval(
    valueK: number,
    maeOrMarginK: number
  ): [number, number] {
    const lower = Math.max(0, (valueK - maeOrMarginK) * 1000);
    const upper = (valueK + maeOrMarginK) * 1000;
    return [lower, upper];
  }

  /** Intervallo diretto dalla response (se vuoi usarlo nel template). */
  getConfidenceIntervalFromResponse(): [number, number] | null {
    if (!this.response) return null;
    const low = this.response.metrics.confidence_low_k * 1000;
    const high = this.response.metrics.confidence_high_k * 1000;
    return [low, high];
  }

  simulatePublish(): void {
    alert('Publishing Simulated');
  }
}
