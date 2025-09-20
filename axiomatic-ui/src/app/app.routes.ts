import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { DashboardComponent } from './pages/dashboard/dashboard.component';
import { LogsComponent } from './pages/logs/logs.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'logs', component: LogsComponent },
  {
    path: 'verify',
    loadComponent: () =>
      import('./components/quick-verify/quick-verify.component').then(
        (m) => m.QuickVerifyComponent
      ),
  },
  { path: '**', redirectTo: '' },
];
