import { Injectable } from '@angular/core';
import {
  HttpClient,
  HttpErrorResponse,
  HttpParams,
} from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { PropertyRequest, PredictionResponse } from '../models/property';
import { firstValueFrom } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  predictProperty(
    data: PropertyRequest,
    publish: boolean = false
  ): Observable<PredictionResponse> {
    const params = new HttpParams().set('publish', publish.toString());
    return this.http
      .post<PredictionResponse>(`${this.baseUrl}/predict/property`, data, {
        params,
      })
      .pipe(catchError(this.handleError));
  }

  getApiLogs(): Promise<any[]> {
    return firstValueFrom(this.http.get<any[]>(`${this.baseUrl}/logs/api`));
  }

  getPublishedLogs(): Promise<any[]> {
    return firstValueFrom(
      this.http.get<any[]>(`${this.baseUrl}/logs/published`)
    );
  }

  getDetailReports(): Promise<any[]> {
    return firstValueFrom(
      this.http.get<any[]>(`${this.baseUrl}/logs/detail_reports`)
    );
  }

  private handleError(error: HttpErrorResponse) {
    console.error('[API ERROR]', error);
    return throwError(() => error);
  }
}
