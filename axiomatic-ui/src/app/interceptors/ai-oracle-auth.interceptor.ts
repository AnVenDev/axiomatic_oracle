import { Injectable, inject } from '@angular/core';
import {
  HttpEvent,
  HttpHandler,
  HttpInterceptor,
  HttpRequest,
  HttpErrorResponse,
} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { ToastService } from '../services/toast.service';

@Injectable()
export class AiOracleAuthInterceptor implements HttpInterceptor {
  private toast = inject(ToastService);

  intercept(
    req: HttpRequest<any>,
    next: HttpHandler
  ): Observable<HttpEvent<any>> {
    let request = req;

    // Auth header (se definita)
    const apiKey: string | undefined =
      (environment as any)?.apiKey || (environment as any)?.API_KEY;
    if (apiKey) {
      request = req.clone({
        setHeaders: { Authorization: `Bearer ${apiKey}` },
      });
    }

    return next.handle(request).pipe(
      catchError((err: any) => {
        if (err instanceof HttpErrorResponse) {
          // rete giù / server irraggiungibile
          if (err.status === 0) {
            this.toast.error(
              'Impossibile contattare il server. Controlla connessione o proxy.'
            );
          }
          // rate limit
          else if (err.status === 429) {
            this.toast.warn('Troppo traffico, riprova tra poco.');
          }
          // body troppo grande
          else if (err.status === 413) {
            this.toast.error('Payload troppo grande. Riduci i dati incollati.');
          }
          // auth mancante/errata
          else if (err.status === 401 || err.status === 403) {
            this.toast.error('Autenticazione richiesta o chiave API invalida.');
          }
          // fallback
          else {
            const msg =
              err.error?.detail || err.message || 'Errore di richiesta.';
            this.toast.error(
              typeof msg === 'string' ? msg : 'Errore di richiesta.'
            );
          }
        }
        return throwError(() => err);
      })
    );
  }
}
