import {
  ApplicationConfig,
  provideZoneChangeDetection,
  importProvidersFrom,
} from '@angular/core';
import { provideRouter } from '@angular/router';
import {
  provideHttpClient,
  withFetch,
  withInterceptors,
} from '@angular/common/http';
import { routes } from './app.routes';
import {
  provideClientHydration,
  withEventReplay,
} from '@angular/platform-browser';
import { provideAnimations } from '@angular/platform-browser/animations';

import {
  MatSnackBarModule,
  MAT_SNACK_BAR_DEFAULT_OPTIONS,
} from '@angular/material/snack-bar';

import { authInterceptor } from './interceptors/auth.interceptor';
import { apiErrorToastInterceptor } from './interceptors/api-error-toast.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),

    importProvidersFrom(MatSnackBarModule),

    // Animazioni per i toast Material
    provideAnimations(),

    // Http + interceptors
    provideHttpClient(
      withFetch(),
      withInterceptors([authInterceptor, apiErrorToastInterceptor])
    ),

    // Router & hydration
    provideRouter(routes),
    provideClientHydration(withEventReplay()),

    // Opzioni default dei toast
    {
      provide: MAT_SNACK_BAR_DEFAULT_OPTIONS,
      useValue: {
        duration: 3500,
        horizontalPosition: 'right',
        verticalPosition: 'bottom',
      },
    },
  ],
};
