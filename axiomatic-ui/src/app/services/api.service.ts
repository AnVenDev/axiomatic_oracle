import { Injectable } from '@angular/core';
import {
  HttpClient,
  HttpErrorResponse,
  HttpParams,
} from '@angular/common/http';
import { firstValueFrom, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { PropertyRequest, PredictionResponseV2 } from '../models/property';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private baseUrl = environment['/api'].target; // ← niente hardcode

  constructor(private http: HttpClient) {}

  health(): Promise<any> {
    return firstValueFrom(
      this.http.get(`${this.baseUrl}/health`).pipe(catchError(this.handleError))
    );
  }

  listModels(asset = 'property'): Promise<any> {
    return firstValueFrom(
      this.http
        .get(`${this.baseUrl}/models/${asset}`)
        .pipe(catchError(this.handleError))
    );
  }

  modelHealth(asset = 'property', task = 'value_regressor'): Promise<any> {
    return firstValueFrom(
      this.http
        .get(`${this.baseUrl}/models/${asset}/${task}/health`)
        .pipe(catchError(this.handleError))
    );
  }

  refreshModel(asset = 'property', task = 'value_regressor'): Promise<any> {
    return firstValueFrom(
      this.http
        .post(`${this.baseUrl}/models/${asset}/${task}/refresh`, {})
        .pipe(catchError(this.handleError))
    );
  }

  predictProperty(
    data: PropertyRequest,
    publish = false
  ): Promise<PredictionResponseV2> {
    const params = new HttpParams().set('publish', String(publish));
    return firstValueFrom(
      this.http
        .post<PredictionResponseV2>(`${this.baseUrl}/predict/property`, data, {
          params,
        })
        .pipe(catchError(this.handleError))
    );
  }

  getApiLogs(): Promise<any[]> {
    return firstValueFrom(
      this.http
        .get<any[]>(`${this.baseUrl}/logs/api`)
        .pipe(catchError(this.handleError))
    );
  }

  getPublishedLogs(): Promise<any[]> {
    return firstValueFrom(
      this.http
        .get<any[]>(`${this.baseUrl}/logs/published`)
        .pipe(catchError(this.handleError))
    );
  }

  getDetailReports(): Promise<any[]> {
    return firstValueFrom(
      this.http
        .get<any[]>(`${this.baseUrl}/logs/detail_reports`)
        .pipe(catchError(this.handleError))
    );
  }

  private handleError(error: HttpErrorResponse) {
    console.error('[API ERROR]', error);
    return throwError(() => error);
  }
}
