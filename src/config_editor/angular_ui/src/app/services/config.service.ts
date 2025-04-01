import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ConfigService {
  private apiUrl = 'http://127.0.0.1:5000';
  constructor(private http: HttpClient) { }

  getProcessingConfig(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/load_processing_config`);
  }

  getTrainingConfig(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/load_training_config`);
  }

  updateProcessingConfig(config: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/update_processing_config`, config);
  }

  updatetrainingConfig(config: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/update_training_config`, config);
  }

  runBacktesting(): Observable<any> {
    return this.http.post(`${this.apiUrl}/run_backtesting`, {});
  }

  shutdownServer(): Observable<any> {
    return this.http.post(`${this.apiUrl}/shutdown`, {});
  }
}
