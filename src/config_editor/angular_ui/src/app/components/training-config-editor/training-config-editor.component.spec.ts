import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainingConfigEditorComponent } from './training-config-editor.component';

describe('TrainingConfigEditorComponent', () => {
  let component: TrainingConfigEditorComponent;
  let fixture: ComponentFixture<TrainingConfigEditorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainingConfigEditorComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TrainingConfigEditorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
