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
  constructor(private configService: ConfigService) {}

  ngOnInit(): void {
    this.configService.getProcessingConfig().subscribe(data => {
      this.config = data.config;
    });
  }

  saveConfig(): void {
    this.configService.updateProcessingConfig(this.config).subscribe(response => {
      alert('✅ Processing Config updated successfully');
    });
  }

}
