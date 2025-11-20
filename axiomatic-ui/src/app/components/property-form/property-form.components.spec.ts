import { ComponentFixture, TestBed } from '@angular/core/testing';
import { PropertyFormComponent } from './property-form.component';
import { PropertyRequest } from '../../models/property';

describe('PropertyFormComponent', () => {
  let component: PropertyFormComponent;
  let fixture: ComponentFixture<PropertyFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PropertyFormComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(PropertyFormComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should emit submit with current value', () => {
    const value = { location: 'Milan' } as PropertyRequest;
    component.value = value;

    const spy = spyOn(component.submit, 'emit');
    component.onSubmit();

    expect(spy).toHaveBeenCalledWith(value);
  });

  it('should emit valueChange on change with a shallow copy', () => {
    const value = { location: 'Rome', size_sqm: 80 } as PropertyRequest;
    component.value = value;

    let emitted: PropertyRequest | undefined;
    component.valueChange.subscribe(v => (emitted = v));

    component.onChange();

    expect(emitted).toBeDefined();
    expect(emitted).toEqual(value);
    // deve essere un nuovo riferimento
    expect(emitted).not.toBe(value);
  });

  it('should toggle advanced section flag', () => {
    expect(component.showAdvanced).toBeFalse();

    component.toggleAdvanced();
    expect(component.showAdvanced).toBeTrue();

    component.toggleAdvanced();
    expect(component.showAdvanced).toBeFalse();
  });
});
