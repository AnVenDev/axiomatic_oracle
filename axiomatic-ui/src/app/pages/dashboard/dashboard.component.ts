import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { ApiService } from '../../services/api.service';
import { PredictionResponse } from '../../models/property';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatTabsModule } from '@angular/material/tabs';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { HttpErrorResponse } from '@angular/common/http';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatInputModule,
    MatTableModule,
    MatPaginatorModule,
    MatTabsModule,
    MatProgressSpinnerModule,
    MatIconModule,
    MatCardModule,
  ],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent {
  jsonInput: string = '';
  response: PredictionResponse | any = null;
  errorMessage: string | null = null;
  selectedLog: any = null;

  constructor(private api: ApiService) {}

  sendRequest(): void {
    this.errorMessage = null;
    this.response = null;

    let payload;
    try {
      payload = JSON.parse(this.jsonInput);
    } catch (e) {
      this.errorMessage = 'Invalid JSON: check syntax.';
      return;
    }

    this.api.predictProperty(payload).subscribe({
      next: (res) => {
        this.errorMessage = null;
        this.response = res;
      },
      error: (err: HttpErrorResponse) => {
        const detail = err.error?.detail || err.error || err.message;
        this.errorMessage = this.translateValidationError(detail);
        this.response = null;
      },
    });
  }

  translateValidationError(detail: any): string {
    if (Array.isArray(detail)) {
      // Standard FastAPI/Pydantic case (array of objects)
      return this.formatErrorList(
        detail.map((e: any) => ({
          field: e.loc?.[1] || 'unknown field',
          message: e.msg,
        }))
      );
    } else if (typeof detail === 'string') {
      // Check if it's a Pydantic error (both "validation errors for" and "validation error for")
      if (
        detail.includes('validation error') &&
        detail.includes('PropertyPredictRequest')
      ) {
        return this.parsePydanticError(detail);
      } else {
        return `Error: ${detail}`;
      }
    }
    return 'Unknown validation error.';
  }

  private parsePydanticError(errorString: string): string {
    const errors: Array<{ field: string; message: string }> = [];

    // FIRST: Handle model_validator errors (format: "Value error, message")
    const valueErrorRegex = /Value error,\s*(.+?)\s*\[type=value_error/g;
    let valueMatch;

    while ((valueMatch = valueErrorRegex.exec(errorString)) !== null) {
      const errorMessage = valueMatch[1].trim();

      if (
        errorMessage.includes('floor must be') &&
        errorMessage.includes('building_floors')
      ) {
        errors.push({
          field: this.translateFieldName('floor'),
          message: 'must be less than the number of building floors',
        });
      } else if (
        errorMessage.includes('energy_class must be one of') ||
        (errorMessage.includes('energy_class') &&
          errorMessage.includes('one of'))
      ) {
        errors.push({
          field: this.translateFieldName('energy_class'),
          message: 'must be one of: A, B, C, D, E, F, G',
        });
      } else {
        // For other model_validator errors not specifically handled
        errors.push({
          field: 'Validation',
          message: errorMessage,
        });
      }
    }

    // AFTER: Regex for standard Pydantic validation errors (format: "field\n  Input should be...")
    const fieldErrorRegex =
      /^(\w+)\s*\n\s*Input should be (.+?)(?=\s*\[type=)/gm;
    let fieldMatch;

    while ((fieldMatch = fieldErrorRegex.exec(errorString)) !== null) {
      const field = fieldMatch[1];
      const rawMessage = fieldMatch[2].trim();

      // Avoid duplicates: if we've already added this field, skip
      const alreadyAdded = errors.some(
        (error) => error.field === this.translateFieldName(field)
      );

      if (!alreadyAdded) {
        const message = this.translateMessage(rawMessage);
        errors.push({
          field: this.translateFieldName(field),
          message,
        });
      }
    }

    return this.formatErrorList(errors);
  }

  private formatErrorList(
    errors: Array<{ field: string; message: string }>
  ): string {
    if (errors.length === 0) return 'Validation error.';

    const header = `Found ${errors.length} validation error${
      errors.length > 1 ? 's' : ''
    }:\n`;

    return (
      header +
      errors
        .map((error, index) => `${index + 1}. ${error.field}: ${error.message}`)
        .join('\n')
    );
  }

  private translateFieldName(field: string): string {
    const translations: { [key: string]: string } = {
      location: 'Location',
      size_m2: 'Size (m²)',
      rooms: 'Rooms',
      bathrooms: 'Bathrooms',
      year_built: 'Year Built',
      floor: 'Floor',
      building_floors: 'Building Floors',
      has_elevator: 'Elevator',
      has_garden: 'Garden',
      has_balcony: 'Balcony',
      garage: 'Garage',
      energy_class: 'Energy Class',
      humidity_level: 'Humidity Level (%)',
      temperature_avg: 'Average Temperature (°C)',
      noise_level: 'Noise Level (dB)',
      air_quality_index: 'Air Quality Index',
      age_years: 'Age (years)',
    };
    return translations[field] || field;
  }

  private translateMessage(message: string): string {
    const translations: { [key: string]: string } = {
      // Messages for data types and parsing
      'a valid string': 'must be a valid string',
      'a valid integer': 'must be a valid integer',
      'a valid number': 'must be a valid number',
      'unable to parse string as an integer': 'must be a valid integer',
      'unable to parse string as a number': 'must be a valid number',

      // Messages for required fields
      'field required': 'is required',

      // For numeric fields with gt (greater than)
      'greater than 0': 'must be greater than 0',

      // For fields with ge (greater than or equal) - correct for your constraints
      'greater than or equal to 0': 'must be at least 0',
      'greater than or equal to 1': 'must be at least 1',
      'greater than or equal to 1800': 'must be at least 1800',

      // For fields with le (less than or equal) - correct for your actual constraints
      'less than or equal to 1': 'must be 0 or 1 (0=No, 1=Yes)',
      'less than or equal to 100': 'cannot exceed 100%',
      'less than or equal to 150': 'cannot exceed 150 dB',
      'less than or equal to 500': 'cannot exceed 500',
    };

    // Check exact translations first
    for (const [eng, translated] of Object.entries(translations)) {
      if (message === eng) {
        return translated;
      }
    }

    // Then check if the message contains any of the keys
    for (const [eng, translated] of Object.entries(translations)) {
      if (message.includes(eng)) {
        return translated;
      }
    }

    // Handle dynamic patterns for current year
    const yearMatch = message.match(/less than or equal to (\d{4})/);
    if (yearMatch) {
      return `cannot be later than ${yearMatch[1]}`;
    }

    // Pattern for combined messages like "a valid integer, unable to parse string as an integer"
    if (
      message.includes('a valid integer') &&
      message.includes('unable to parse')
    ) {
      return 'must be a valid integer';
    }

    if (
      message.includes('a valid number') &&
      message.includes('unable to parse')
    ) {
      return 'must be a valid number';
    }

    // If no translation found, return the original message
    return message;
  }

  simulatePublish(): void {
    alert('Publishing Simulated');
  }
}
