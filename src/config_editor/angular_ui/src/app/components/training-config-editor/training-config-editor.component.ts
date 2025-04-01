import { Component } from '@angular/core';
import { ConfigService } from '../../services/config.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-training-config-editor',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './training-config-editor.component.html',
  styleUrl: './training-config-editor.component.css'
})
export class TrainingConfigEditorComponent {
  config: any = {
    splitter: {
      min_training_months: 12,
      testing_months: 3
    },
    clustering_processor: {
      max_clusters: 30,
      default_cluster_size: 4
    },
    models_params: {
      MLModels: {
        ElasticNet: { enabled: true, alpha: 0, l1_ratio: 0 },
        KNeighborsRegressor: { enabled: false, n_neighbors: 5, weights: "distance" },
        RandomForest: { enabled: false, n_estimators: 100, max_depth: 5 },
        XGBoost: { enabled: false, n_estimators: 100, max_depth: 5 }
      },
      MLP: { enabled: true, activation_function: "relu", solver: "adam" }
    }
  };

  isInvalid = false; // Flag to indicate whether the config is invalid
  errorMessage = '';
  isRunning = false; // Flag to indicate whether backtesting is running
  allModelsDisabled = false; // Flag to indicate whether all models are disabled

  constructor(private configService: ConfigService) {}

  ngOnInit(): void {
    this.configService.getTrainingConfig().subscribe(data => {
      this.config = data.config;
      this.validateConfig();
    });
  }

  validateConfig(): void {
    this.errorMessage = '';

    if (this.config.splitter.min_training_months <= 0) {
      this.errorMessage = 'Minimum training months must be greater than 0.';
    } else if (this.config.splitter.testing_months <= 0) {
      this.errorMessage = 'Testing months must be greater than 0.';
    } else if (this.config.clustering_processor.max_clusters <= 0) {
      this.errorMessage = 'Max clusters must be greater than 0.';
    } else if (this.config.clustering_processor.default_cluster_size <= 0) {
      this.errorMessage = 'Default cluster size must be greater than 0.';
    }

    this.isInvalid = this.errorMessage !== '';
  }

  saveConfig(): void {
    if (!this.isInvalid) {
      this.configService.updateTrainingConfig(this.config).subscribe(response => {
        alert('✅ Training Config updated successfully');
      });
    }
  }

  runBacktesting(): void {
    this.isRunning = true;

    this.configService.runBacktesting().subscribe({
      next: (response) => {
        alert('✅ Backtesting completed.');
        this.isRunning = false;
      },
      error: (error) => {
        alert('❌ Error while running backtesting: ' + error.message);
        this.isRunning = false;
      }
    });
  }

  shutdown(): void {
    this.configService.shutdownServer().subscribe({
      next: (response) => {
        alert('🛑 Server shutting down...');
        window.close();
      },
      error: (error) => {
        alert('❌ Error shutting down server: ' + error.message);
      }
    });
  }

  checkModels() {
    this.allModelsDisabled = !(
      this.config.models_params.MLModels.ElasticNet.enabled ||
      this.config.models_params.MLModels.KNeighborsRegressor.enabled ||
      this.config.models_params.MLModels.RandomForest.enabled ||
      this.config.models_params.MLModels.XGBoost.enabled ||
      this.config.models_params.MLP.enabled      
    );
  }


}
