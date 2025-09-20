import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { PropertyRequest } from '../../models/property';

@Component({
  standalone: true,
  selector: 'app-property-form',
  imports: [
    CommonModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
  ],
  templateUrl: './property-form.component.html',
  styleUrls: ['./property-form.component.scss'],
})
export class PropertyFormComponent {
  @Input() value: PropertyRequest = {} as PropertyRequest;
  @Input() disabled = false;
  @Input() loading = false;
  @Output() valueChange = new EventEmitter<PropertyRequest>();

  showAdvanced = false;

  /** submit verso il parent (Dashboard) */
  @Output() submit = new EventEmitter<PropertyRequest>();

  onSubmit() {
    // Se vuoi fare normalizzazione qui, fallo prima dell’emit
    this.submit.emit(this.value);
  }

  onChange() {
    // emetto una copia per evitare mutazioni in-place
    this.valueChange.emit({ ...(this.value || {}) });
  }

  toggleAdvanced() {
    this.showAdvanced = !this.showAdvanced;
  }

  energyClasses = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
}
