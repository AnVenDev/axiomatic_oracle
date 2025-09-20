import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { environment } from '../../../environments/environment';

export type WorkflowKey = 'request' | 'publish' | 'verify' | 'download';
export type WorkflowStatus =
  | 'todo'
  | 'pending'
  | 'success'
  | 'error'
  | 'skipped';

export interface WorkflowStep {
  key: WorkflowKey;
  label: string;
  status: WorkflowStatus;
  meta?: { txid?: string; note?: string; url?: string };
}

@Component({
  selector: 'app-workflow-tracker',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './workflow-tracker.component.html',
  styleUrls: ['./workflow-tracker.component.scss'],
})
export class WorkflowTrackerComponent {
  @Input() title = 'Workflow';
  @Input() steps: WorkflowStep[] = [
    { key: 'request', label: 'Send to Oracle', status: 'todo' },
    { key: 'publish', label: 'Publish on-chain', status: 'todo' },
    { key: 'verify', label: 'Verify', status: 'todo' },
    { key: 'download', label: 'Download JSON', status: 'todo' },
  ];

  /** Override opzionale della base explorer; se non fornito usa environment */
  @Input() explorerBase: string | null = null;

  urlFor(s: WorkflowStep): string | null {
    if (s?.meta?.url) return s.meta.url!;
    const txid = s?.meta?.txid;
    if (!txid) return null;
    const net = (environment as any).defaultNetwork || 'testnet';
    const base =
      (this.explorerBase ?? (environment as any).explorers?.[net]) || null;
    return base ? `${base}${txid}` : null;
  }
}
