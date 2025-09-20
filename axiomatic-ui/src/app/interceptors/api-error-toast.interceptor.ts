// src/app/interceptors/api-error-toast.interceptor.ts
import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

export const apiErrorToastInterceptor: HttpInterceptorFn = (req, next) => {
  const snack = inject(MatSnackBar);

  return next(req).pipe(
    catchError((err: HttpErrorResponse) => {
      let msg = 'Errore di rete';

      if (err.status === 401 || err.status === 403) {
        msg = 'API key mancante o non valida';
      } else if (err.status === 429) {
        msg = 'Troppo traffico, riprova tra poco';
      } else if (err.status === 413) {
        msg = 'Payload troppo grande';
      } else if (err.error?.detail) {
        msg =
          typeof err.error.detail === 'string'
            ? err.error.detail
            : JSON.stringify(err.error.detail);
      } else if (err.message) {
        msg = err.message;
      }

      snack.open(msg, 'OK', {
        duration: 4000,
        horizontalPosition: 'right',
        verticalPosition: 'top',
      });

      return throwError(() => err);
    })
  );
};
