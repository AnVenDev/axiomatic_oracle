// src/app/pages/home/home.component.ts
import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { environment } from '../../../environments/environment';

type Range = '24h' | '7d' | '30d';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatButtonToggleModule,
    MatSnackBarModule,
    MatTooltipModule,
  ],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent {
  // Base per curl: se vuoto, usa il proxy '/api'
  apiBase = environment.apiBase ?? '';
  apiBaseNormalized =
    (environment.apiBase && environment.apiBase.trim()) || '/api';
  // Rete / explorer da environment (override globale)
  readonly network = (environment as any).defaultNetwork || 'testnet';
  readonly explorerBase: string | null =
    ((environment as any).explorers?.[this.network] as string | null) ?? null;

  readonly requestId =
    (globalThis.crypto as any)?.randomUUID?.() ??
    Math.random().toString(36).slice(2);

  // KPI state (placeholder: cablali alla tua API appena pronta)
  range: Range = '24h';
  kpi = {
    attestations: { v: '—', sub: 'today / 7d / total' },
    unique: { v: '—', sub: 'lifetime' },
    finality: { v: '—', sub: 'publish → confirmed' },
  };

  // service status (mock → collega ai tuoi endpoint /health e /ready)
  status = {
    api: { label: 'API: OK', ok: true },
    indexer: { label: 'Indexer: OK', ok: true },
    network: { label: 'Network: —', ok: false }, // verrà sincronizzata sotto
  };

  curlSnippet = `curl -X POST ${this.apiBaseNormalized}/predict/property?attestation_only=1 \\
  -H "Authorization: Bearer <API_KEY>" \\
  -H "X-Request-ID: ${this.requestId}" \\
  -H "Content-Type: application/json" \\
  -d '{ "...": "sample payload" }'`;

  constructor(private router: Router, private snack: MatSnackBar) {
    this.syncStatus();
  }

  private syncStatus() {
    const labelMap: Record<string, string> = {
      mainnet: 'Mainnet',
      testnet: 'Testnet',
      betanet: 'Betanet',
      sandbox: 'Sandbox',
    };
    const nice = labelMap[(this.network || '').toLowerCase()] || this.network;
    const ok = !!this.explorerBase;
    this.status.network = { label: `Network: ${nice}`, ok };
  }

  // CTA
  tryDemo(): void {
    this.router.navigate(['/dashboard'], { queryParams: { demo: 1 } });
  }
  goToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }
  goToLogs(): void {
    this.router.navigate(['/logs']);
  }

  // Copy helper
  async copyCurl(): Promise<void> {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(this.curlSnippet);
      } else {
        const ta = document.createElement('textarea');
        ta.value = this.curlSnippet;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      this.snack.open('Snippet copiato', 'OK', { duration: 1800 });
    } catch {
      this.snack.open('Impossibile copiare', 'OK', { duration: 2000 });
    }
  }

  // KPI time range
  setRange(r: Range) {
    this.range = r;
    // TODO: carica KPI reali da API e aggiorna this.kpi.*
  }
}
