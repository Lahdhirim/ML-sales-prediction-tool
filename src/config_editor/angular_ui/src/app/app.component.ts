import { Component } from '@angular/core';
import { ProcessingConfigEditorComponent } from './components/processing-config-editor/processing-config-editor.component';
import { TrainingConfigEditorComponent } from './components/training-config-editor/training-config-editor.component';

@Component({
  selector: 'app-root',
  imports: [ProcessingConfigEditorComponent, TrainingConfigEditorComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'angular_ui';
}
