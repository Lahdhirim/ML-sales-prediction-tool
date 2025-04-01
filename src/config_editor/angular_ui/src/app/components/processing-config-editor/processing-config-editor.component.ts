import { Component } from '@angular/core';
import { ConfigService } from '../../services/config.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-processing-config-editor',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './processing-config-editor.component.html',
  styleUrl: './processing-config-editor.component.css'
})
export class ProcessingConfigEditorComponent {
  config = {
    nb_months_to_predict: 3,
    max_window_size: 12,
    min_window_size: 3,
    max_lag: 12,
    min_lag: 3
  };

  isInvalid = false; // Flag to indicate whether the config is invalid
  errorMessage = '';

  constructor(private configService: ConfigService) {}

  ngOnInit(): void {
    this.configService.getProcessingConfig().subscribe(data => {
      this.config = data.config;
      this.validateConfig();
    });
  }

  validateConfig(): void {
    this.errorMessage = '';

    if (this.config.nb_months_to_predict <= 0) {
      this.errorMessage = 'Number of months to predict must be greater than 0.';
    } else if (this.config.max_window_size <= 0 || this.config.min_window_size <= 0) {
      this.errorMessage = 'Window size values must be greater than 0.';
    } else if (this.config.max_lag <= 0 || this.config.min_lag <= 0) {
      this.errorMessage = 'Lag values must be greater than 0.';
    } else if (this.config.min_window_size > this.config.max_window_size) {
      this.errorMessage = 'Minimum window size cannot be greater than maximum window size.';
    } else if (this.config.min_lag > this.config.max_lag) {
      this.errorMessage = 'Minimum lag cannot be greater than maximum lag.';
    }

    this.isInvalid = this.errorMessage !== '';
  }

  saveConfig(): void {
    if (!this.isInvalid) {
      this.configService.updateProcessingConfig(this.config).subscribe(response => {
        alert('✅ Processing Config updated successfully');
      });
    }
  }

}
