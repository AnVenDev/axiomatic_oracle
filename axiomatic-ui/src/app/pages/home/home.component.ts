import { Component } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { Router } from '@angular/router';

@Component({
  selector: 'app-home',
  imports: [MatCardModule],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent {
  constructor(private router: Router) {}

  goToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }

  // TODO:
  // 1. Hero più compatta
  // Il blocco mat-card.hero-card potrebbe essere leggermente più stretto (padding e max-width: 800px) per migliorare leggibilità da mobile o viewport stretti.

  // 2. Migliora il contrasto nei paragrafi
  // Alcuni testi sono poco leggibili su sfondo scuro (vedi "Affidabile, estendibile, scalabile."):
  // Usare un --color-muted più chiaro o cambiare opacity (es. 0.85) per migliorare leggibilità.

  // 3. ancore o scorrimento dolce
  // I pulsanti come "Scopri di più" possono scrollare automaticamente alla sezione successiva (Come funziona) → UX più fluida.

  // 4. Responsive check su mobile
  // VerificaRE padding orizzontali su mobile: la sezione features-section, faq-section, demo-section potrebbero beneficiare di padding-inline: 1rem.

  // 5. Footer più ricco
  // Aggiungere logo mini o claim
  // Aggiungere social (GitHub, X, Email)
  // Link legali o privacy policy (anche solo placeholder)
}
