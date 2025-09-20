import { HttpInterceptorFn } from '@angular/common/http';
import { environment } from '../../environments/environment';

const needsAuthFor = (url: string) =>
  url.startsWith('/predict') ||
  url.startsWith('/models') ||
  url.startsWith('/logs') ||
  url.startsWith('/verify');

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const apiKey = localStorage.getItem(environment.apiKeyStorageKey)?.trim();
  if (apiKey && needsAuthFor(req.url)) {
    req = req.clone({ setHeaders: { Authorization: `Bearer ${apiKey}` } });
  }
  return next(req);
};
