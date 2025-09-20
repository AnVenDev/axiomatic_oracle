import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';

@Component({
  standalone: true,
  selector: 'app-quick-verify',
  imports: [CommonModule, FormsModule, MatButtonModule, MatInputModule],
  templateUrl: './quick-verify.component.html',
  styleUrls: ['./quick-verify.component.scss'],
})
export class QuickVerifyComponent implements OnChanges {
  /** txid precompilabile dalla dashboard (ultima publish) */
  @Input() defaultTxid = '';

  /** emesso quando l’utente chiede la verifica via JSON */
  @Output() verifyFromJson = new EventEmitter<void>();

  /** emesso (opzionale) se vuoi intercettare la submit esternamente */
  @Output() submitTxid = new EventEmitter<string>();

  txid = '';

  ngOnChanges(changes: SimpleChanges) {
    if ('defaultTxid' in changes && this.defaultTxid && !this.txid) {
      // non autoverifichiamo: lo rendiamo disponibile al bottone "Verify last TX"
    }
  }

  submit(): void {
    const t = (this.txid || '').trim();
    if (!t) return;
    this.submitTxid.emit(t);
  }

  useLastTx(): void {
    if (!this.defaultTxid) return;
    this.txid = this.defaultTxid;
    this.submit();
  }
}
