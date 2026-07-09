# Feature Importance Notes

## Important Internal Features

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

## Important Contextual Features

### Ambient_Temperature
Represents the surrounding environmental temperature that may influence machine performance.

### Load_Density
Represents the operational load conditions affecting machine stress.

## Engineered Features

### Temp_Difference
Difference between process temperature and air temperature.

### Power_Index
Calculated using:
Power_Index = Rotational Speed × Torque

## Expected Impact

The contextual and engineered features may improve:
- Failure prediction accuracy
- Recall score
- F1 Score

These features will be evaluated during the ablation study.
