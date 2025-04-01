import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ProcessingConfigEditorComponent } from './processing-config-editor.component';

describe('ProcessingConfigEditorComponent', () => {
  let component: ProcessingConfigEditorComponent;
  let fixture: ComponentFixture<ProcessingConfigEditorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ProcessingConfigEditorComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ProcessingConfigEditorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
