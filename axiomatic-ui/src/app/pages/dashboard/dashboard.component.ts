import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, MatButtonModule, MatInputModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent {
  jsonInput: string = '';
  response: any = null;

  constructor(private http: HttpClient) {}

  sendRequest(): void {
    let payload;
    try {
      payload = JSON.parse(this.jsonInput);
    } catch (e) {
      alert('JSON non valido!');
      return;
    }

    this.http
      .post('http://localhost:8000/predict/property?publish=false', payload)
      .subscribe({
        next: (res) => (this.response = res),
        error: (err) => {
          alert('Errore dalla API: ' + (err.error?.detail || err.message));
          this.response = null;
        },
      });
  }

  simulatePublish(): void {
    alert('⚠️ Publishing simulato (da integrare con Algorand)');
  }
}
